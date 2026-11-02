#!/usr/bin/env python3
"""Fit the detector at every candidate layer to select the best one."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from architecture import describe
from data import (
    build_prompt_split,
    load_alpaca_control_text,
    load_hard_eval_sets,
    load_wikitext_test,
    load_wikitext_train,
)
from evaluation import binary_metrics
from refusal_circuit import (
    blind_norms_to_dimension,
    fit_constant_direction,
    forward,
    marker_sequences,
    pad_left,
    select_reserved_dimensions,
    silence_dimension,
    train_probe,
    window_text,
)
from run_refusal_circuit import resolve_chat_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--examples-per-class", type=int, default=400)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--orbench-safe-count", type=int, default=3000)
    parser.add_argument("--prompt-token-negatives", type=int, default=100_000)
    parser.add_argument("--probe-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--control-negative-tokens", type=int, default=20_000)
    parser.add_argument("--wikitext-train-negatives", type=int, default=40_000)
    parser.add_argument("--control-sequence-length", type=int, default=96)
    parser.add_argument("--heldout-tokens", type=int, default=20_000)
    parser.add_argument("--heldout-sequence-length", type=int, default=128)
    parser.add_argument("--negative-margin", type=float, default=2.0)
    parser.add_argument("--detector-scale", type=float, default=100.0)
    parser.add_argument("--harm-trigger-gain", type=float, default=2.0)
    parser.add_argument("--harm-trigger-cap", type=float, default=10.0)
    parser.add_argument("--prompt-prefix", default="User: ")
    parser.add_argument("--prompt-suffix", default="\nAssistant")
    parser.add_argument("--prompt-marker", default=":")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layers", default=None, help="comma-separated; default all eligible"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "sweep_detector_layer.pt"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


class AllLayerCapture:
    """
    Returns the MLP input `n = g⊙x/rms(x)` entering every block,
    for one forward pass.
    """

    def __init__(self, architecture) -> None:
        self.architecture = architecture
        self.features: dict[int, torch.Tensor] = {}
        self.handles = [
            architecture.mlp_gate(block).register_forward_pre_hook(self._mlp_hook(index))
            for index, block in enumerate(architecture.blocks)
        ]

    def _mlp_hook(self, index):
        def hook(_module, inputs) -> None:
            self.features[index] = inputs[0].detach().float()
        return hook

    def run(self, input_ids, attention_mask):
        self.features.clear()
        forward(self.architecture, input_ids, attention_mask)
        return self.features

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def capture_sequences(
    capture, tokenizer, sequences, batch_size, device, layers, max_other_tokens=None
):
    """
    Keep all marker positions; subsample the rest uniformly across all sequences
    to about `max_other_tokens` tokens in total.
    """
    total_other = sum(len(sequence) - 1 for sequence in sequences)
    keep_fraction = (
        1.0
        if max_other_tokens is None
        else min(1.0, max_other_tokens / max(total_other, 1))
    )
    generator = torch.Generator(device=device).manual_seed(0)
    out = {layer: [] for layer in layers}
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        input_ids, attention_mask = pad_left(tokenizer, batch, device)
        features = capture.run(input_ids, attention_mask)
        keep = attention_mask.bool()
        if keep_fraction < 1.0:
            keep &= torch.rand(keep.shape, generator=generator, device=device) < keep_fraction
        keep[:, -1] = True
        for layer in layers:
            for row in range(len(batch)):
                out[layer].append(features[layer][row][keep[row]].half())
    return out


@torch.no_grad()
def capture_markers(
    capture, tokenizer, sequences, batch_size, device, layers
):
    """Features at the marker positions, grouped by layer."""
    out = {layer: [] for layer in layers}
    for start in range(0, len(sequences), batch_size):
        input_ids, attention_mask = pad_left(
            tokenizer, sequences[start : start + batch_size], device
        )
        features = capture.run(input_ids, attention_mask)
        for layer in layers:
            out[layer].append(features[layer][:, -1].half())
    return {layer: torch.cat(out[layer]) for layer in layers}


@torch.no_grad()
def capture_text(
    capture, tokenizer, text, max_tokens, sequence_length, batch_size, device, layers
):
    """features at every token of running text, grouped by layer."""
    windows = window_text(tokenizer, text, max_tokens, sequence_length)
    feats = {layer: [] for layer in layers}
    for start in range(0, len(windows), batch_size):
        input_ids = windows[start : start + batch_size].to(device)
        attention_mask = torch.ones_like(input_ids)
        features = capture.run(input_ids, attention_mask)
        for layer in layers:
            feats[layer].append(features[layer].reshape(-1, features[layer].shape[-1]).half())
    return {layer: torch.cat(feats[layer]) for layer in layers}


def trigger_written(n, weight, direction, scale, gain, cap):
    """What the installed detector pair writes into `trigger` at these tokens."""
    gate = scale * (n @ weight)
    up = n @ direction
    rising = F.silu(gate) * up
    capping = F.silu(gate - cap * up) * up
    return gain * (rising - capping)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = model.to(device).eval()
    architecture = describe(model)
    num_layers = len(architecture.blocks)

    prefix, suffix, marker_token = resolve_chat_format(
        tokenizer, args.prompt_prefix, args.prompt_suffix, args.prompt_marker
    )
    print(f"chat format: prefix={prefix!r} suffix={suffix!r} marker={tokenizer.decode([marker_token])!r}")
    wrap = lambda p: prefix + p + suffix

    split = build_prompt_split(
        args.data_dir, args.examples_per_class, args.val_fraction, args.orbench_safe_count
    )
    harmful_train = [wrap(p) for p in split.harmful_train]
    benign_train = [wrap(p) for p in split.benign_train]
    harmful_val = [wrap(p) for p in split.harmful_val]
    benign_val = [wrap(p) for p in split.benign_val]
    print(f"harmful_train={len(harmful_train)} benign_train={len(benign_train)}")

    alpaca_text = load_alpaca_control_text(args.data_dir)
    trigger_dimension, latch_dimension = select_reserved_dimensions(
        architecture, tokenizer, alpaca_text, 2, args.control_sequence_length, args.batch_size, device
    )
    print(f"selected_dimensions=[{trigger_dimension}, {latch_dimension}]")
    for dimension in (trigger_dimension, latch_dimension):
        silence_dimension(architecture, dimension, range(0, num_layers))
        blind_norms_to_dimension(architecture, dimension, range(0, num_layers))
        with torch.no_grad():
            architecture.embed_in.weight[:, dimension] = 0.0
            if not architecture.tied_embeddings:
                architecture.unembed.weight[:, dimension] = 0.0
    reserved = [trigger_dimension, latch_dimension]

    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(range(0, num_layers - 2))
    )
    capture = AllLayerCapture(architecture)
    try:
        seqs = lambda prompts: marker_sequences(
            tokenizer, prompts, marker_token, args.max_length
        )
        per_class = args.prompt_token_negatives // 2
        harmful_train_cap = capture_sequences(
            capture, tokenizer, seqs(harmful_train), args.batch_size, device, layers, per_class
        )
        benign_train_cap = capture_sequences(
            capture, tokenizer, seqs(benign_train), args.batch_size, device, layers, per_class
        )
        val_cap = capture_sequences(
            capture, tokenizer, seqs(harmful_val + benign_val), args.batch_size, device, layers
        )
        alpaca_cap = capture_text(
            capture, tokenizer, alpaca_text, args.control_negative_tokens,
            args.control_sequence_length, args.batch_size, device, layers
        )
        wtrain_cap = capture_text(
            capture, tokenizer, load_wikitext_train(args.data_dir), args.wikitext_train_negatives,
            args.control_sequence_length, args.batch_size, device, layers
        )
        wtest_cap = capture_text(
            capture, tokenizer, load_wikitext_test(args.data_dir), args.heldout_tokens,
            args.heldout_sequence_length, args.batch_size, device, layers
        )
        hard_sets = load_hard_eval_sets(args.data_dir, exclude=split.seen)
        hard_cap = {
            name: tuple(
                capture_markers(
                    capture, tokenizer, seqs([wrap(p) for p in prompts]),
                    args.batch_size, device, layers
                )
                if prompts else None
                for prompts in (hard_harmful, hard_benign)
            )
            for name, (hard_harmful, hard_benign) in hard_sets.items()
        }
    finally:
        capture.close()

    val_labels = torch.cat((
        torch.zeros(len(harmful_val), dtype=torch.long, device=device),
        torch.full((len(benign_val),), -1, dtype=torch.long, device=device),
    ))
    hard_names = list(hard_cap)
    for name, (hard_harmful, hard_benign) in hard_sets.items():
        print(f"hard_set[{name}] harmful={len(hard_harmful)} benign={len(hard_benign)}")
    print(
        "layer | val recall / FPR | score gap | v@marker mean cv | v@other q99.9 absmax | "
        "gate>=0 wt-test | trigger>1 wt-test | trigger max wt-test | trigger@harmful median min | trigger@benign max | "
        + " | ".join(f"{name} recall / FPR" for name in hard_names)
    )
    results = []
    for layer in layers:
        def marker_rows(cap):
            return torch.stack([f[-1] for f in cap[layer]]).float()

        def other_rows(cap):
            return torch.cat([f[:-1] for f in cap[layer]]).float()

        harmful_marker = marker_rows(harmful_train_cap)
        benign_marker = marker_rows(benign_train_cap)
        val_marker = marker_rows(val_cap)
        other_tokens = torch.cat(
            (other_rows(harmful_train_cap), other_rows(benign_train_cap),
             alpaca_cap[layer].float(), wtrain_cap[layer].float())
        )
        for tensor in (harmful_marker, benign_marker, val_marker, other_tokens):
            tensor[:, reserved] = 0.0

        train_features = torch.cat((harmful_marker, benign_marker, other_tokens))
        train_labels = torch.cat(
            (torch.zeros(len(harmful_marker), dtype=torch.long),
             torch.full((len(benign_marker) + len(other_tokens),), -1, dtype=torch.long))
        )
        sizes = (len(harmful_marker), len(benign_marker), len(other_tokens))
        sample_weight = torch.cat([torch.full((n,), 1.0 / n) for n in sizes])
        weight, _ = train_probe(
            train_features, train_labels, val_marker, val_labels, 1, args.probe_steps,
            negative_margin=args.negative_margin, sample_weight=sample_weight,
        )
        weight = weight.squeeze(0)
        weight[reserved] = 0.0
        direction, dstats = fit_constant_direction(
            torch.cat((harmful_marker, benign_marker)), others=other_tokens
        )
        direction[reserved] = 0.0

        with torch.no_grad():
            val_scores = val_marker @ weight
            metrics = binary_metrics(val_scores >= 0, (val_labels == 0).long())
            harmful_scores = val_scores[val_labels == 0]
            benign_scores = val_scores[val_labels != 0]
            gap = float(
                (torch.quantile(harmful_scores, 0.01) - torch.quantile(benign_scores, 0.99))
                / harmful_scores.std().clamp_min(1e-8)
            )
            wt_f = wtest_cap[layer].float()
            wt_f[:, reserved] = 0.0
            gate_rate = float(((wt_f @ weight) >= 0).float().mean())
            wt_trigger = trigger_written(
                wt_f, weight, direction, args.detector_scale,
                args.harm_trigger_gain, args.harm_trigger_cap
            )
            marker_trigger = trigger_written(
                val_marker, weight, direction, args.detector_scale,
                args.harm_trigger_gain, args.harm_trigger_cap
            )
            harmful_trigger = marker_trigger[val_labels == 0]
            benign_trigger = marker_trigger[val_labels != 0]
            hard_rates = {}
            for name in hard_names:
                for role, cap in zip(("recall", "fpr"), hard_cap[name]):
                    if cap is None:
                        hard_rates[f"{name}_{role}"] = float("nan")
                        continue
                    rows_ = cap[layer].float()
                    rows_[:, reserved] = 0.0
                    hard_rates[f"{name}_{role}"] = float(((rows_ @ weight) >= 0).float().mean())
        row = {
            **hard_rates,
            "layer": layer,
            "recall": metrics["harmful_recall"], "fpr": metrics["benign_false_positive_rate"],
            "gap": gap, "v_mean": dstats["mean"], "v_cv": dstats["cv"],
            "v_other_q999": dstats["others_q999"], "v_other_absmax": dstats["others_absmax"],
            "gate_rate": gate_rate, "trigger_rate": float((wt_trigger.abs() > 1).float().mean()),
            "trigger_max": float(wt_trigger.abs().max()),
            "harmful_trigger_median": float(harmful_trigger.median()), "harmful_trigger_min": float(harmful_trigger.min()),
            "benign_trigger_max": float(benign_trigger.abs().max()),
        }
        results.append(row)
        print(
            f"{layer:5d} | {row['recall']:6.1%} / {row['fpr']:5.1%} | {gap:8.2f} | "
            f"{row['v_mean']:.3f} {row['v_cv']:.3f} | {row['v_other_q999']:.3f} {row['v_other_absmax']:6.2f} | "
            f"{gate_rate:8.4%} | {row['trigger_rate']:8.4%} | {row['trigger_max']:8.2f} | "
            f"{row['harmful_trigger_median']:6.2f} {row['harmful_trigger_min']:6.2f} | {row['benign_trigger_max']:6.2f} | "
            + " | ".join(
                f"{hard_rates[f'{name}_recall']:6.1%} / {hard_rates[f'{name}_fpr']:6.1%}"
                for name in hard_names
            ),
            flush=True,
        )
    torch.save(results, args.output)


if __name__ == "__main__":
    main()
