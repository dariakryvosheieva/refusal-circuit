#!/usr/bin/env python3
"""End-to-end driver to train and evaluate a model with a built-in refusal circuit."""


from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from architecture import describe
from data import (
    build_prompt_split,
    load_alpaca_control_text,
    load_alpaca_conversations,
    load_wikitext_train,
    load_wikitext_test,
    load_hard_eval_sets,
)
from evaluation import (
    evaluate_language_model,
    print_metrics,
    binary_metrics,
    exact_refusal_rate,
    evaluate_out_of_distribution,
    generate_responses,
    save_generations,
)
from refusal_circuit import (
    REFUSAL_TEXT,
    select_reserved_dimensions,
    reserve_coordinates,
    marker_sequences,
    window_tokens,
    measure_pre_mlp_rms,
    extract_detector_features,
    extract_token_level_negatives,
    train_probe,
    fit_constant_direction,
    install_trigger_detector_neurons,
    install_latch_attention_head,
    measure_latch_ratio,
    extract_output_features,
    report_circuit_signals,
    extract_unlatched_control_features,
    install_output_neurons,
)


CONFIG_ARGS = (
    "detector_layer", "latch_layer", "head_index", "trigger_dimension",
    "latch_dimension", "query_gain", "latch_gain", "output_gain", "detector_scale",
    "output_scale", "harm_trigger_gain", "harm_trigger_cap", "negative_margin",
    "marker_threshold", "orbench_safe_count", "wikitext_negatives",
    "alpaca_response_negatives", "prompt_prefix", "prompt_suffix",
)


def resolve_chat_format(
    tokenizer,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    prompt_marker: str | None = None,
) -> tuple[str, str, int]:
    """
    Returns `(prefix, suffix, marker)` with `prefix` and `suffix` as text
    and `marker` as a token id.
    Explicit arguments override the corresponding defaults.
    """
    prefix = prompt_prefix or "User: "
    suffix = prompt_suffix or "\nAssistant"
    marker_text = prompt_marker or ":"
    marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
    if len(marker_ids) != 1:
        raise ValueError(f"marker {marker_text!r} is not a single token: {marker_ids}")
    marker = marker_ids[0]
    # The suffix followed by the marker must tokenize as suffix tokens + marker.
    joined = tokenizer.encode(suffix + marker_text, add_special_tokens=False)
    if joined != tokenizer.encode(suffix, add_special_tokens=False) + [marker]:
        raise ValueError(f"{suffix + marker_text!r} does not tokenize as suffix + marker")
    return prefix, suffix, marker


def build_response_sequences(
    tokenizer,
    conversations: list[tuple[str, str]],
    marker_token: int,
    max_length: int,
) -> tuple[list[list[int]], list[int]]:
    """
    Builds sequences of the form "chat-formatted prompt + marker + response";
    returns the sequences and, for each, the index of the first response token.
    """
    separator = "" if tokenizer.decode([marker_token]).endswith("\n") else " "
    sequences: list[list[int]] = []
    starts: list[int] = []
    for prompt, response in conversations:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) + [marker_token]
        response_ids = tokenizer.encode(separator + response, add_special_tokens=False)
        response_ids = response_ids[: max(1, max_length - len(prompt_ids))]
        sequence = (prompt_ids + response_ids)[-max_length:]
        sequences.append(sequence)
        starts.append(len(sequence) - len(response_ids))
    return sequences, starts


def response_negatives(
    architecture, tokenizer, args, marker_token, wrap, device
) -> torch.Tensor:
    """
    Detector-layer features of assistant-response tokens in the chat format,
    i.e. the model's own output after the marker.
    """
    conversations = [
        pair for pair in load_alpaca_conversations(args.data_dir)
    ]
    random.Random(args.seed).shuffle(conversations)
    conversations = [
        (wrap(user), response)
        for user, response in conversations[: args.response_conversations]
    ]
    sequences, starts = build_response_sequences(
        tokenizer, conversations, marker_token, args.max_length
    )
    _, tokens = extract_detector_features(
        architecture, tokenizer, sequences, args.detector_layer, args.batch_size,
        device, starts=starts, other_cap=args.alpaca_response_negatives,
    )
    print(f"response_negatives={len(tokens)} tokens from {len(conversations)} conversations")
    return tokens


def build_marker_prefix_sequences(
    tokenizer,
    prompts: list[str],
    marker_token: int,
    refusal_tokens: list[int],
    max_length: int,
) -> tuple[list[list[int]], torch.Tensor]:
    """
    Builds sequences of the form "prompt + marker + every refusal prefix",
    including the complete refusal; returns the sequences along with their lengths.
    """
    sequences: list[list[int]] = []
    states: list[int] = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        for state in range(len(refusal_tokens) + 1):
            suffix = [marker_token] + refusal_tokens[:state]
            sequences.append(prompt_ids[-(max_length - len(suffix)) :] + suffix)
            states.append(state)
    return sequences, torch.tensor(states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # --- global ---

    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(__file__).parent / "marker_gated_refusal.pt",
        help="path to save the model checkpoint",
    )
    parser.add_argument(
        "--save-generations", type=Path, default=None,
        help="path to write the edited model's full responses to the OOD sets "
        "for judge-based scoring of any refusal, not just the exact one",
    )

    # --- forward pass ---

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--generation-tokens", type=int, default=96)

    # --- collateral evaluation ---
    
    parser.add_argument("--collateral-tokens", type=int, default=20_000)
    parser.add_argument("--collateral-sequence-length", type=int, default=128)

    # --- neuron training ---

    parser.add_argument(
        "--examples-per-class", type=int, default=400,
        help="for training detector neurons",
    )
    parser.add_argument(
        "--output-prompts-per-class", type=int, default=80,
        help="for training output neurons",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--orbench-safe-count", type=int, default=3000,
        help="OR-Bench safe prompts (benign, on dangerous topics) in the detector's training data",
    )
    parser.add_argument("--probe-steps", type=int, default=1000)

    parser.add_argument(
        "--prompt-token-negatives", type=int, default=100_000,
        help="cap on non-marker prompt tokens used as detector negatives",
    )
    parser.add_argument(
        "--alpaca-instruction-negatives", type=int, default=20_000,
        help="tokens of Alpaca instructions added to the negatives",
    )
    parser.add_argument(
        "--wikitext-negatives", type=int, default=40_000,
        help="tokens of WikiText-2 (train) added to the negatives",
    )
    parser.add_argument(
        "--alpaca-response-negatives", type=int, default=100_000,
        help="tokens of Alpaca responses added to the negatives",
    )

    parser.add_argument("--response-conversations", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=160)

    # --- chat format ---

    parser.add_argument(
        "--prompt-prefix",
        default='User: ',
        help="text before user content",
    )
    parser.add_argument(
        "--prompt-suffix",
        default='\nAssistant',
        help="text between user content and marker",
    )
    parser.add_argument(
        "--prompt-marker",
        type=str,
        default=':',
        help="position to extract harmfulness feature from",
    )

    # --- circuit settings ---

    parser.add_argument("--detector-layer", type=int, default=None)
    parser.add_argument("--latch-layer", type=int, default=None)
    parser.add_argument("--head-index", type=int, default=3)
    parser.add_argument("--trigger-dimension", type=int, default=None)
    parser.add_argument("--latch-dimension", type=int, default=None)
    parser.add_argument(
        "--control-sequence-length", type=int, default=96,
        help="for identifying least active dimensions",
    )

    parser.add_argument("--query-gain", type=float, default=10.0)
    parser.add_argument("--latch-gain", type=float, default=0.5)
    parser.add_argument("--output-gain", type=float, default=10.0)

    parser.add_argument("--detector-scale", type=float, default=100.0)
    parser.add_argument("--output-scale", type=float, default=8.0)
    
    parser.add_argument("--harm-trigger-gain", type=float, default=2.0)
    parser.add_argument("--harm-trigger-cap", type=float, default=10.0)
    
    parser.add_argument(
        "--negative-margin", type=float, default=2.0,
        help="make detector negatives score below -margin, not merely below 0",
    )
    parser.add_argument(
        "--marker-threshold", type=float, default=0.0,
        help="shift the detector's decision threshold at the marker by this much",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    refusal_tokens = tokenizer.encode(REFUSAL_TEXT, add_special_tokens=False)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = model.to(device).eval()
    architecture = describe(model)
    num_layers = len(architecture.blocks)
    
    prompt_prefix, prompt_suffix, marker_token = resolve_chat_format(
        tokenizer, args.prompt_prefix, args.prompt_suffix, args.prompt_marker
    )
    print(
        f"chat format: prefix={prompt_prefix!r} suffix={prompt_suffix!r} "
        f"marker={tokenizer.decode([marker_token])!r}"
    )

    split = build_prompt_split(
        args.data_dir, args.examples_per_class, args.val_fraction, args.orbench_safe_count
    )
    wrap = lambda p: prompt_prefix + p + prompt_suffix
    harmful_train = [wrap(p) for p in split.harmful_train]
    harmful_val = [wrap(p) for p in split.harmful_val]
    benign_train = [wrap(p) for p in split.benign_train]
    benign_val = [wrap(p) for p in split.benign_val]
    print(
        f"harmful_train={len(harmful_train)} benign_train={len(benign_train)} "
        f"val={len(harmful_val)}+{len(benign_val)}"
    )

    alpaca_text = load_alpaca_control_text(args.data_dir)
    wikitext_train = load_wikitext_train(args.data_dir)
    wikitext_test = load_wikitext_test(args.data_dir)

    collateral_ids = tokenizer.encode(
        wikitext_test, add_special_tokens=False, return_tensors="pt"
    ).squeeze(0)[: args.collateral_tokens]

    def collateral(name: str) -> dict[str, float] | None:
        """WikiText-2 test perplexity of the model as it currently stands."""
        metrics = evaluate_language_model(
            model, collateral_ids, args.collateral_sequence_length, args.batch_size, device
        )
        print_metrics(name, metrics)
        return metrics
    
    baseline = collateral("baseline")

    if args.detector_layer is None:
        args.detector_layer = round(2 * num_layers / 3) - 1
    if args.latch_layer is None:
        args.latch_layer = args.detector_layer + 1
    if not args.detector_layer < args.latch_layer <= num_layers - 2:
        raise ValueError("need detector_layer < latch layer <= num_layers - 2")

    # Reserve the circuit's coordinates
    if None in (args.trigger_dimension, args.latch_dimension):
        args.trigger_dimension, args.latch_dimension = select_reserved_dimensions(
            architecture, tokenizer, alpaca_text, 2, args.control_sequence_length, args.batch_size, device
        )
    reserved = [args.trigger_dimension, args.latch_dimension]
    print(f"selected_dimensions={reserved}")
    silenced_entries = reserve_coordinates(architecture, args.trigger_dimension, args.latch_dimension, args.latch_layer)
    reserved_only = collateral("reserved_coordinates_only")

    # Typical residual scale at the final block, used to size the output
    # neurons' push relative to the residual.
    final_rms = measure_pre_mlp_rms(
        architecture, tokenizer,
        marker_sequences(tokenizer, harmful_train[:64] + benign_train[:64], marker_token, args.max_length),
        num_layers - 1, args.batch_size, device,
    )
    print(f"typical_rms final_layer={final_rms:.2f}")
    
    # --- Train the detector ---
    def prompt_features(prompts: list[str], other_cap: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detector-layer MLP input at the marker of each prompt + marker,
        and at about `other_cap` of the prompts' other tokens.
        """
        sequences = marker_sequences(tokenizer, prompts, marker_token, args.max_length)
        return extract_detector_features(
            architecture, tokenizer, sequences, args.detector_layer,
            args.batch_size, device, other_cap=other_cap
        )
    def text_negatives(text: str, max_tokens: int) -> torch.Tensor:
        """Detector-layer MLP input at every token of running text."""
        return extract_token_level_negatives(
            architecture, tokenizer, text, max_tokens, args.control_sequence_length,
            args.batch_size, device, args.detector_layer,
        )
    def labels(positives: int, negatives: int) -> torch.Tensor:
        """Detector labels: state 0 for the positives, -1 (negative) for the rest."""
        return torch.cat((
            torch.zeros(positives, dtype=torch.long, device=device),
            torch.full((negatives,), -1, dtype=torch.long, device=device),
        ))
    
    train_marker, prompt_tokens = prompt_features(harmful_train + benign_train, args.prompt_token_negatives)
    harmful_marker, benign_marker = train_marker[: len(harmful_train)], train_marker[len(harmful_train) :]
    val_marker, _ = prompt_features(harmful_val + benign_val)
    val_marker_labels = labels(len(harmful_val), len(benign_val))

    negatives = [prompt_tokens]
    if args.alpaca_instruction_negatives > 0:
        negatives.append(text_negatives(alpaca_text, args.alpaca_instruction_negatives))
    if args.wikitext_negatives > 0:
        negatives.append(text_negatives(wikitext_train, args.wikitext_negatives))
    if args.alpaca_response_negatives > 0:
        negatives.append(response_negatives(architecture, tokenizer, args, marker_token, wrap, device))
    other_tokens = torch.cat(negatives)
    del negatives, prompt_tokens

    for tensor in (harmful_marker, benign_marker, val_marker, other_tokens):
        tensor[:, reserved] = 0.0
    
    group_sizes = (len(harmful_marker), len(benign_marker), len(other_tokens))
    detector_weight, _ = train_probe(
        torch.cat((harmful_marker, benign_marker, other_tokens)),
        labels(group_sizes[0], sum(group_sizes[1:])),
        val_marker,
        val_marker_labels,
        1,
        args.probe_steps,
        negative_margin=args.negative_margin,
        sample_weight=torch.cat([torch.full((size,), 1.0 / size, device=device) for size in group_sizes]),
    )
    constant_direction, direction_stats = fit_constant_direction(
        torch.cat((harmful_marker, benign_marker)), others=other_tokens
    )
    constant_direction[reserved] = 0.0
    print(
        f"constant_direction at markers: mean={direction_stats['mean']:.3f} "
        f"cv={direction_stats['cv']:.3f} min={direction_stats['min']:.3f} "
        f"max={direction_stats['max']:.3f} | on other tokens: "
        f"mean={direction_stats['others_mean']:.4f} "
        f"|.|q99.9={direction_stats['others_q999']:.3f} "
        f"absmax={direction_stats['others_absmax']:.3f}"
    )
    if args.marker_threshold:
        detector_weight = detector_weight - args.marker_threshold * constant_direction.unsqueeze(0)
        print(f"marker_threshold={args.marker_threshold:+.2f} applied as w - threshold*v")
    detector_weight[:, reserved] = 0.0
    weight = detector_weight.squeeze(0)

    with torch.no_grad():
        train_metrics = binary_metrics(
            train_marker @ weight >= 0, labels(len(harmful_marker), len(benign_marker)) == 0
        )
        val_metrics = binary_metrics(val_marker @ weight >= 0, (val_marker_labels == 0).long())
        other_fire_rate = ((other_tokens @ weight) >= 0).float().mean().item()
    other_count = len(other_tokens)
    del other_tokens

    # --- Install detector, trigger, and latch head ---
    detector_circuit = install_trigger_detector_neurons(
        architecture, args.detector_layer, detector_weight, constant_direction, args.trigger_dimension,
        args.detector_scale, args.harm_trigger_gain, args.harm_trigger_cap,
    )
    latch_circuit = install_latch_attention_head(
        architecture, args.latch_layer, args.head_index, args.trigger_dimension, args.latch_dimension,
        args.query_gain, args.latch_gain,
    )
    # The output neurons' up branch will read the latch; measure how big it is
    # relative to the final-layer RMS on latched training prompts.
    latch_statistics = measure_latch_ratio(
        architecture, tokenizer,
        marker_sequences(tokenizer, harmful_train[:128], marker_token, args.max_length),
        args.latch_dimension, args.batch_size, device,
    )
    latch_ratio = latch_statistics["ratio_median"]
    print(
        f"latch_ratio rms/latch median={latch_ratio:.2f} cv={latch_statistics['ratio_cv']:.3f} "
        f"latch median={latch_statistics['latch_median']:.3f} "
        f"latched={latch_statistics['latched']}/{latch_statistics['total']} "
        f"unlatched absmax={latch_statistics['latch_unlatched_absmax']:.4f}"
    )
    report_circuit_signals(architecture, tokenizer, harmful_val[:32], benign_val[:32], args, marker_token, device)
    latch_only = collateral("detector_and_latch_head_only")

    # --- Train output neurons ---
    def output_features(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        return extract_output_features(
            architecture, tokenizer, sequences, args.batch_size,
            device, args.detector_layer, detector_weight
        )
    output_prompts = harmful_train[: args.output_prompts_per_class] + benign_train[: args.output_prompts_per_class]
    train_sequences, train_states = build_marker_prefix_sequences(
        tokenizer, output_prompts, marker_token, refusal_tokens, args.max_length
    )
    val_sequences, val_states = build_marker_prefix_sequences(
        tokenizer, harmful_val + benign_val, marker_token, refusal_tokens, args.max_length
    )
    permutation = torch.randperm(len(train_sequences))
    train_sequences = [train_sequences[index] for index in permutation.tolist()]
    train_states = train_states[permutation].to(device)
    train_features, train_latched = output_features(train_sequences)
    train_labels = torch.where(train_latched, train_states, -1)
    control_features, alpaca_statistics = extract_unlatched_control_features(
        architecture, tokenizer, alpaca_text, args.alpaca_instruction_negatives,
        args.control_sequence_length, args.batch_size, device,
        args.detector_layer, detector_weight
    )
    train_features = torch.cat((train_features, control_features))
    train_labels = torch.cat((train_labels, labels(0, len(control_features))))
    val_output_features, val_latched = output_features(val_sequences)
    val_labels = torch.where(val_latched, val_states.to(device), -1)
    # Whether or not the detector fired is determined by the `up` branch;
    # the `gate` assumes it did and only needs to classify which token of the
    # refusal sequence was just emitted.
    terminal = len(refusal_tokens)
    latched_train = train_labels >= 0
    latched_val = val_labels >= 0
    gate_train = train_features[latched_train].clone()
    gate_val = val_output_features[latched_val].clone()
    gate_train[:, reserved] = 0.0
    gate_val[:, reserved] = 0.0
    gate_train_labels = torch.where(train_labels == terminal, -1, train_labels)[latched_train]
    gate_val_labels = torch.where(val_labels == terminal, -1, val_labels)[latched_val]
    output_weight, output_metrics = train_probe(
        gate_train, gate_train_labels, gate_val, gate_val_labels,
        len(refusal_tokens), args.probe_steps,
    )
    output_weight[:, reserved] = 0.0
    output_metrics["terminal_silence"] = output_metrics["benign_specificity"]
    # informational only: when unlatched the up branch is exactly zero
    with torch.no_grad():
        unlatched_gate = val_output_features[~latched_val] @ output_weight.T
    output_metrics["benign_specificity"] = (
        float((unlatched_gate.amax(dim=1) < 0).float().mean())
        if (~latched_val).any()
        else float("nan")
    )
    transition_circuit = install_output_neurons(
        architecture, refusal_tokens, output_weight, args.latch_dimension,
        latch_ratio, final_rms, args.output_scale, args.output_gain,
    )

    # --- Validation ---
    def exact_refusal(prompts: list[str]) -> tuple[float, float, list[str]]:
        return exact_refusal_rate(
            model, tokenizer, prompts, refusal_tokens, marker_token,
            args.batch_size, args.max_length, device,
        )

    harmful_exact, harmful_stopped, harmful_samples = exact_refusal(harmful_val)
    benign_exact, _, benign_samples = exact_refusal(benign_val)
    print(
        f"marker={tokenizer.decode([marker_token])!r} "
        f"detector_layer={args.detector_layer} latch_layer={args.latch_layer} "
        f"head={args.head_index} trigger_dim={args.trigger_dimension} latch_dim={args.latch_dimension}"
    )
    print(f"other_token_fire_rate={other_fire_rate:.4%} other_tokens={other_count}")
    print(
        f"detector_train_recall={train_metrics['harmful_recall']:.4%} "
        f"detector_train_fpr={train_metrics['benign_false_positive_rate']:.4%} "
        f"detector_val_recall={val_metrics['harmful_recall']:.4%} "
        f"detector_val_fpr={val_metrics['benign_false_positive_rate']:.4%} "
        f"detector_val_accuracy={val_metrics['accuracy']:.4%}"
    )
    print(
        f"alpaca_token_fire_rate={alpaca_statistics['token_fire_rate']:.4%} "
        f"alpaca_window_latch_rate={alpaca_statistics['window_latch_rate']:.4%}"
    )
    print(
        f"state_probe_accuracy={output_metrics['state_accuracy']:.4%} "
        f"terminal_silence={output_metrics['terminal_silence']:.4%} "
        f"unlatched_specificity={output_metrics['benign_specificity']:.4%} "
        f"latched_train_sequences={int(train_latched.sum())}/{len(train_latched)}"
    )
    print(f"harmful_exact_refusal={harmful_exact:.4%} followed_by_eos={harmful_stopped:.4%}")
    print(f"benign_exact_refusal={benign_exact:.4%}")
    print(
        f"detector_selected_entries={detector_circuit['selected_entries']} "
        f"attention_selected_entries={latch_circuit['selected_entries']} "
        f"silenced_entries={silenced_entries} "
        f"transition_selected_entries={transition_circuit['selected_entries']}"
    )
    print(f"harmful_samples={harmful_samples}")
    print(f"benign_samples={benign_samples}")

    # --- Out-of-distribution evaluation ---
    hard_eval: dict[str, dict] = {}
    hard_sets = load_hard_eval_sets(args.data_dir, exclude=split.seen)
    hard_eval = evaluate_out_of_distribution(
        hard_sets, wrap, lambda prompts: prompt_features(prompts)[0], lambda prompts: exact_refusal(prompts)[:2],
        detector_weight, constant_direction, reserved,
    )
    if args.save_generations is not None:
        save_generations(
            args.save_generations, hard_sets, wrap,
            lambda prompts: generate_responses(
                model, tokenizer, prompts, marker_token, args.generation_tokens,
                args.batch_size, args.max_length, device,
            ),
        )

    collateral_metrics = collateral("marker_gated_circuit")
    wikitext_statistics = None
    if collateral_metrics is not None:
        print(
            f"perplexity_change="
            f"{(collateral_metrics['perplexity'] / baseline['perplexity'] - 1):.2%} "
            f"latch_only_ppl_change="
            f"{(latch_only['perplexity'] / baseline['perplexity'] - 1):.2%} "
            f"accuracy_change="
            f"{collateral_metrics['next_token_accuracy'] - baseline['next_token_accuracy']:.2%}"
        )
        _, wikitext_statistics = extract_unlatched_control_features(
            architecture, tokenizer, wikitext_test, args.collateral_tokens,
            args.collateral_sequence_length, args.batch_size, device, args.detector_layer, detector_weight,
        )
        print(
            f"wikitext_token_fire_rate={wikitext_statistics['token_fire_rate']:.4%} "
            f"wikitext_window_latch_rate={wikitext_statistics['window_latch_rate']:.4%}"
        )

    torch.save(
        {
            "model": args.model,
            "marker_token": marker_token,
            "refusal_tokens": refusal_tokens,
            "detector_weight": detector_weight.cpu(),
            "probe_weight": output_weight.cpu(),
            "detector_val_metrics": val_metrics,
            "other_token_fire_rate": other_fire_rate,
            "alpaca_statistics": alpaca_statistics,
            "output_metrics": output_metrics,
            "harmful_exact_refusal": harmful_exact,
            "benign_exact_refusal": benign_exact,
            "hard_eval": hard_eval,
            "baseline_collateral": baseline,
            "reserved_only_collateral": reserved_only,
            "latch_only_collateral": latch_only,
            "collateral": collateral_metrics,
            "wikitext_statistics": wikitext_statistics,
            "circuit_config": {
                **{name: getattr(args, name) for name in CONFIG_ARGS},
                "architecture": architecture.model_type,
                "latch_ratio": latch_ratio,
                "constant_direction": constant_direction.cpu(),
                "final_rms": final_rms,
            },
        },
        args.checkpoint,
    )
    print(f"saved={args.checkpoint}")


if __name__ == "__main__":
    main()
