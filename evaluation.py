"""Evaluation and generation scripts."""


import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from refusal_circuit import (
    REFUSAL_TEXT,
    marker_sequences,
    window_tokens,
    pad_left,
)


def binary_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Accuracy, recall, and FPR for binary classification (harmful vs benign)."""
    prediction = prediction.bool()
    target = target.bool()
    true_positive = int((prediction & target).sum())
    false_positive = int((prediction & ~target).sum())
    true_negative = int((~prediction & ~target).sum())
    false_negative = int((~prediction & target).sum())
    return {
        "accuracy": (true_positive + true_negative) / len(target),
        "harmful_recall": true_positive / max(1, true_positive + false_negative),
        "benign_false_positive_rate": false_positive
        / max(1, false_positive + true_negative),
    }


@torch.no_grad()
def greedy_continuations(
    model,
    tokenizer,
    prompts: list[str],
    marker_token: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
    **generate_kwargs,
):
    """
    Greedy continuations of `prompt + marker`, one batch at a time
    (yields the new tokens only).
    """
    for start in range(0, len(prompts), batch_size):
        sequences = marker_sequences(
            tokenizer, prompts[start : start + batch_size], marker_token, max_length
        )
        input_ids, attention_mask = pad_left(tokenizer, sequences, device)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            **generate_kwargs,
        )
        yield generated[:, input_ids.shape[1] :]


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    marker_token: int,
    max_new_tokens: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> list[str]:
    """
    Greedy responses of length up to `max_new_tokens` after `prompt + marker`
    for all prompts (returned as text).
    """
    return [
        tokenizer.decode(row, skip_special_tokens=True).strip()
        for continuation in greedy_continuations(
            model, tokenizer, prompts, marker_token, batch_size, max_length, device,
            max_new_tokens=max_new_tokens,
        )
        for row in continuation
    ]


@torch.no_grad()
def exact_refusal_rate(
    model,
    tokenizer,
    prompts: list[str],
    refusal_tokens: list[int],
    marker_token: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[float, float, list[str]]:
    """
    Fraction of prompts whose first `len(refusal_tokens)` greedy tokens
    are exactly the refusal sequence, the fraction of those refusals whose
    next token is end-of-text (the circuit fell silent and the model
    stopped), plus up to three decoded samples.
    """
    exact = 0
    stopped = 0
    samples: list[str] = []
    target = torch.tensor(refusal_tokens, device=device)
    for continuation in greedy_continuations(
        model, tokenizer, prompts, marker_token, batch_size, max_length, device,
        max_new_tokens=len(refusal_tokens) + 1, min_new_tokens=len(refusal_tokens) + 1,
        eos_token_id=None,
    ):
        refused = (continuation[:, : len(refusal_tokens)] == target).all(dim=1)
        exact += int(refused.sum())
        stopped += int((refused & (continuation[:, len(refusal_tokens)] == tokenizer.eos_token_id)).sum())
        samples += [
            tokenizer.decode(row[: len(refusal_tokens)].tolist())
            for row in continuation[: 3 - len(samples)]
        ]
    return exact / len(prompts), stopped / max(1, exact), samples


def evaluate_out_of_distribution(
    hard_sets: dict[str, tuple[list[str], list[str]]],
    wrap: Callable[[str], str],
    marker_features: Callable[[list[str]], torch.Tensor],
    exact_refusal: Callable[[list[str]], tuple[float, float]],
    detector_weight: torch.Tensor,
    constant_direction: torch.Tensor,
    reserved: list[int],
) -> dict[str, dict[str, dict]]:
    """
    Detector decisions, exact refusal rates and the share of exact refusals
    followed by end-of-text on all OOD test sets, plus the same decisions
    under shifted thresholds relative to the installed one (the detector's
    operating curve).
    """
    hard_eval: dict[str, dict[str, dict]] = {}
    for name, (hard_harmful, hard_benign) in hard_sets.items():
        hard_eval[name] = {}
        parts = []
        for role, prompts in (("harmful", hard_harmful), ("benign", hard_benign)):
            if not prompts:
                hard_eval[name][role] = {"count": 0, "detector_rate": float("nan"), "exact_refusal": float("nan")}
                continue
            marker = marker_features([wrap(p) for p in prompts])
            marker[:, reserved] = 0.0
            score = marker @ detector_weight.squeeze(0)
            refused, stopped = exact_refusal([wrap(p) for p in prompts])
            stats = {
                "count": len(prompts),
                "detector_rate": float((score >= 0).float().mean()),
                "exact_refusal": refused,
                "followed_by_eos": stopped,
                "score": score,
                "constant": marker @ constant_direction,
            }
            hard_eval[name][role] = stats
            parts.append(
                f"{role}(n={stats['count']}) detector={stats['detector_rate']:.2%} "
                f"exact_refusal={stats['exact_refusal']:.2%} followed_by_eos={stopped:.2%}"
            )
        print(f"hard_eval[{name}] " + " | ".join(parts))
    columns = [
        (name, role) for name in hard_eval for role in ("harmful", "benign")
        if hard_eval[name][role]["count"]
    ]
    print("threshold_shift | " + " | ".join(f"{n}/{r[0]}" for n, r in columns))
    for shift in (-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
        rates = [
            float(((stats["score"] - shift * stats["constant"]) >= 0).float().mean())
            for stats in (hard_eval[name][role] for name, role in columns)
        ]
        print(f"{shift:+15.1f} | " + " | ".join(f"{rate:9.1%}" for rate in rates))
    for name, role in columns:
        del hard_eval[name][role]["score"], hard_eval[name][role]["constant"]
    return hard_eval


def save_generations(
    path: Path,
    hard_sets: dict[str, tuple[list[str], list[str]]],
    wrap: Callable[[str], str],
    generate: Callable[[list[str]], list[str]],
) -> None:
    """
    Full responses of the circuit-installed model to the OOD sets,
    for scoring free-form refusals with `judge_generations.py`.
    Also reports the circuit firing at a marker-like token inside the answer:
    refusal text not at the start.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for name, (hard_harmful, hard_benign) in hard_sets.items():
            for role, prompts in (("harmful", hard_harmful), ("benign", hard_benign)):
                if not prompts:
                    continue
                responses = generate([wrap(p) for p in prompts])
                mid_response = 0
                for prompt, response in zip(prompts, responses):
                    exact = response.startswith(REFUSAL_TEXT)
                    mid_response += int(not exact and REFUSAL_TEXT in response)
                    handle.write(json.dumps({
                        "set": name, "kind": role, "prompt": prompt, "response": response,
                        "exact_refusal": exact,
                    }) + "\n")
                print(
                    f"generations[{name}/{role}] n={len(prompts)} "
                    f"mid_response_refusal={mid_response / len(prompts):.2%}"
                )
    print(f"saved_generations={path}")


@torch.no_grad()
def evaluate_language_model(
    model,
    token_ids: torch.Tensor,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Next-token NLL / perplexity / accuracy on windows of `token_ids`."""
    sequences = window_tokens(token_ids, sequence_length)
    total_nll = 0.0
    total_correct = 0
    total_tokens = 0

    for start in range(0, len(sequences), batch_size):
        input_ids = sequences[start : start + batch_size].to(device)
        logits = model(input_ids=input_ids).logits
        predicted_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
        targets = input_ids[:, 1:].reshape(-1)
        total_nll += float(
            F.cross_entropy(predicted_logits, targets, reduction="sum")
        )
        total_correct += int((predicted_logits.argmax(dim=-1) == targets).sum())
        total_tokens += targets.numel()

    average_nll = total_nll / total_tokens
    return {
        "nll": average_nll,
        "perplexity": math.exp(average_nll),
        "next_token_accuracy": total_correct / total_tokens,
        "evaluated_tokens": float(total_tokens),
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name}: nll={metrics['nll']:.6f} "
        f"perplexity={metrics['perplexity']:.4f} "
        f"next_token_accuracy={metrics['next_token_accuracy']:.4%} "
        f"tokens={int(metrics['evaluated_tokens'])}"
    )
