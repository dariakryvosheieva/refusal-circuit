"""Shared building blocks for the refusal circuit."""


from __future__ import annotations

import csv
import io
import json
import math
import random
import urllib.request
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F


REFUSAL_TEXT = "I'm sorry, I cannot assist with this request."


# --------------------------------------------------------------------------
# Forward pass and feature extraction
# --------------------------------------------------------------------------


def pad_left(
    tokenizer, sequences: list[list[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), width), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        input_ids[row, -len(sequence) :] = values
        attention_mask[row, -len(sequence) :] = 1
    return input_ids, attention_mask


def positions_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Rotary positions starting at the first non-pad token, as `generate` does."""
    positions = attention_mask.long().cumsum(dim=-1) - 1
    return positions.masked_fill(attention_mask == 0, 1)


def forward(architecture, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """One forward pass of a left-padded batch with `generate`'s positions."""
    return architecture.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=positions_from_mask(attention_mask),
    )


@torch.no_grad()
def forward_batches(architecture, tokenizer, sequences: list[list[int]], batch_size: int, device):
    """
    Run `sequences` through the model in left-padded batches; yields each
    batch's attention mask (forward hooks collect whatever is needed).
    """
    for start in range(0, len(sequences), batch_size):
        input_ids, attention_mask = pad_left(tokenizer, sequences[start : start + batch_size], device)
        forward(architecture, input_ids, attention_mask)
        yield attention_mask


def marker_sequences(
    tokenizer, prompts: list[str], marker_token: int, max_length: int
) -> list[list[int]]:
    """Tokenize and add the marker token to the end of each prompt."""
    budget = max_length - 1
    return [
        tokenizer.encode(prompt, add_special_tokens=False)[-budget:] + [marker_token]
        for prompt in prompts
    ]


def window_tokens(token_ids: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Cut a token stream into full non-overlapping windows, shape `(n, L)`."""
    usable = (token_ids.numel() // sequence_length) * sequence_length
    return token_ids[:usable].view(-1, sequence_length)


def window_text(tokenizer, text: str, max_tokens: int, sequence_length: int) -> torch.Tensor:
    """The first `max_tokens` tokens of running text, in windows."""
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    return window_tokens(token_ids.squeeze(0)[:max_tokens], sequence_length)


def _feature_hook(architecture, block, store: dict, name: str):
    """
    Capture the MLP's input `n = g ⊙ x / rms(x)`, i.e. the RMSNorm output.

    The probes are fitted on exactly what the installed neurons will see, so
    an installed gate row `w` produces the pre-activation `w·n` with no
    further scaling.
    """

    def mlp_hook(_module, inputs) -> None:
        store[name] = inputs[0].detach().float()

    return [architecture.mlp_gate(block).register_forward_pre_hook(mlp_hook)]


@torch.no_grad()
def run_with_captures(
    architecture,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    detector_layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass returning the detector-layer and final-layer MLP inputs."""
    captured: dict[str, torch.Tensor] = {}
    handles = _feature_hook(architecture, architecture.blocks[detector_layer], captured, "detector")
    handles += _feature_hook(architecture, architecture.blocks[-1], captured, "final")
    try:
        forward(architecture, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    return captured["detector"], captured["final"]


@torch.no_grad()
def extract_detector_features(
    architecture,
    tokenizer,
    sequences: list[list[int]],
    layer_index: int,
    batch_size: int,
    device: torch.device,
    starts: list[int] | None = None,
    other_cap: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inputs to the detector-layer MLP:
    * `last` - the marker (last token) of each sequence;
    * `others` - a uniform sample of about `other_cap` rows in total
      from the sequences' other tokens.
    """
    starts = starts or [0] * len(sequences)
    total = sum(len(sequence) - 1 - start for sequence, start in zip(sequences, starts))
    fraction = min(1.0, other_cap / max(total, 1))
    last: list[torch.Tensor] = []
    others: list[torch.Tensor] = []
    for begin in range(0, len(sequences), batch_size):
        batch = sequences[begin : begin + batch_size]
        input_ids, attention_mask = pad_left(tokenizer, batch, device)
        features, _ = run_with_captures(architecture, input_ids, attention_mask, layer_index)
        last.append(features[:, -1])
        keep = attention_mask.bool()
        keep[:, -1] = False
        if any(starts):
            index = attention_mask.cumsum(dim=1) - 1  # position among the non-pad tokens
            keep &= index >= torch.tensor(starts[begin : begin + batch_size], device=device)[:, None]
        if fraction < 1.0:
            keep &= torch.rand(keep.shape, device=device) < fraction
        others.append(features[keep])
    return torch.cat(last), torch.cat(others)


@torch.no_grad()
def extract_token_level_negatives(
    architecture,
    tokenizer,
    text: str,
    max_tokens: int,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    layer_index: int,
) -> torch.Tensor:
    """MLP input at `layer_index` for every token of running text."""
    windows = window_text(tokenizer, text, max_tokens, sequence_length)
    collected: list[torch.Tensor] = []
    for start in range(0, len(windows), batch_size):
        input_ids = windows[start : start + batch_size].to(device)
        features, _ = run_with_captures(architecture, input_ids, torch.ones_like(input_ids), layer_index)
        collected.append(features.reshape(-1, architecture.hidden_size))
    return torch.cat(collected)


@torch.no_grad()
def extract_output_features(
    architecture,
    tokenizer,
    sequences: list[list[int]],
    batch_size: int,
    device: torch.device,
    detector_layer: int,
    detector_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs to the final-layer MLP plus whether the detector fired anywhere."""
    weight = detector_weight.squeeze(0)
    features: list[torch.Tensor] = []
    latched: list[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        input_ids, attention_mask = pad_left(
            tokenizer, sequences[start : start + batch_size], device
        )
        detector_inputs, final_inputs = run_with_captures(
            architecture, input_ids, attention_mask, detector_layer
        )
        logits = detector_inputs @ weight
        logits[~attention_mask.bool()] = -float("inf")
        latched.append(logits.amax(dim=1) >= 0)
        features.append(final_inputs[:, -1, :])
    return torch.cat(features), torch.cat(latched)


@torch.no_grad()
def extract_unlatched_control_features(
    architecture,
    tokenizer,
    text: str,
    max_tokens: int,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    detector_layer: int,
    detector_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Final-layer features of non-latched tokens.

    Also returns the rate at which the detector's gate is non-negative on
    these tokens and the fraction of windows containing such a token. This
    is a pessimistic false positive estimate: the trigger the circuit writes
    is also multiplied by the `up` branch, which reads ~0 off the marker.
    """
    weight = detector_weight.squeeze(0)
    windows = window_text(tokenizer, text, max_tokens, sequence_length)
    kept: list[torch.Tensor] = []
    fired_tokens = 0
    latched_windows = 0
    for start in range(0, len(windows), batch_size):
        input_ids = windows[start : start + batch_size].to(device)
        attention_mask = torch.ones_like(input_ids)
        detector_inputs, final_inputs = run_with_captures(
            architecture, input_ids, attention_mask, detector_layer
        )
        fired = (detector_inputs @ weight) >= 0
        fired_tokens += int(fired.sum())
        latched_windows += int(fired.any(dim=1).sum())
        latched_so_far = fired.long().cummax(dim=1).values.bool()
        kept.append(final_inputs[~latched_so_far])
    statistics = {
        "token_fire_rate": fired_tokens / max(1, windows.numel()),
        "window_latch_rate": latched_windows / max(1, len(windows)),
        "windows": len(windows),
    }
    return torch.cat(kept), statistics


# --------------------------------------------------------------------------
# Detector training
# --------------------------------------------------------------------------


def train_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    states: int,
    steps: int,
    negative_margin: float = 0.0,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Train a linear probe (to be installed into the detector neurons) to
    separate marker tokens in a harmful prompt from everything else.

    The probe is trained so that positives score above zero, while negatives
    score below `negative_margin`.

    `sample_weight` (one entry per training row) replaces the default
    positive/negative class balancing, e.g. to keep a small set of hard
    negatives from being drowned by a large set of easy ones.
    """
    device = train_features.device
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)
    std = train_features.std(dim=0).clamp_min(1e-4)
    train_scaled = train_features / std
    test_scaled = test_features / std
    train_targets = torch.zeros(len(train_labels), states, device=device)
    positive = train_labels >= 0
    train_targets[positive, train_labels[positive]] = 1.0
    if sample_weight is None:
        positive_count = train_targets.sum(dim=0).clamp_min(1.0)
        positive_weight = (len(train_targets) - positive_count) / positive_count
        sample_weight = torch.ones(len(train_targets), 1, device=device)
    else:
        positive_weight = torch.ones(states, device=device)
        sample_weight = sample_weight.to(device).reshape(-1, 1) / sample_weight.mean()

    weight = torch.zeros(states, train_features.shape[1], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([weight], lr=3e-2, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = train_scaled @ weight.T
        shifted = logits + negative_margin * (1.0 - train_targets)
        loss = F.binary_cross_entropy_with_logits(
            shifted, train_targets, pos_weight=positive_weight, reduction="none"
        )
        loss = (loss * sample_weight).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = test_scaled @ weight.T
        labelled = test_labels >= 0
        predicted = logits.argmax(dim=1)
        state_accuracy = (
            (predicted[labelled] == test_labels[labelled])
            & (logits[labelled].amax(dim=1) >= 0)
        ).float().mean().item() if labelled.any() else float("nan")
        benign_specificity = (
            logits[~labelled].amax(dim=1) < 0
        ).float().mean().item() if (~labelled).any() else float("nan")
        folded = (weight.detach() / std.unsqueeze(0))
    
    return folded, {
        "state_accuracy": state_accuracy,
        "benign_specificity": benign_specificity,
    }


def fit_constant_direction(
    features: torch.Tensor,
    others: torch.Tensor | None = None,
    ridge: float = 1e-2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Direction `v` with `v·n` ~1 on `features` and ~0 on `others`.

    Note that here, `features` is marker-token representations across both
    harmful and benign prompts, while `others` is representations of non-marker
    tokens. This is different from positives/negatives because we want the
    marker token's identity to act as a bias.
    """
    gram = features.T.double() @ features.double()
    target = features.double().sum(dim=0)
    if others is not None and len(others):
        scale = len(features) / len(others)
        gram += scale * (others.T.double() @ others.double())
    dim = gram.shape[0]
    gram += ridge * (gram.trace() / dim) * torch.eye(dim, dtype=torch.float64, device=gram.device)
    direction = torch.linalg.solve(gram, target).float()
    projection = features @ direction
    stats = {
        "mean": float(projection.mean()),
        "cv": float(projection.std() / projection.mean().abs()),
        "min": float(projection.min()),
        "max": float(projection.max()),
    }
    if others is not None and len(others):
        other_projection = others @ direction
        stats["others_mean"] = float(other_projection.mean())
        stats["others_absmax"] = float(other_projection.abs().max())
        stats["others_q999"] = float(torch.quantile(other_projection.abs(), 0.999))
    return direction, stats


# --------------------------------------------------------------------------
# Circuit elements
# --------------------------------------------------------------------------


@torch.no_grad()
def select_reserved_dimensions(
    architecture, tokenizer, text: str, count: int, sequence_length: int, batch_size: int, device
) -> list[int]:
    """
    Select the `count` least-active residual coordinates, i.e., those with the
    lowest squared contribution `(x_j·g_j)^2` to the MLP inputs, averaged over
    all blocks and all tokens of `text`.
    """
    usage = torch.zeros(architecture.hidden_size, dtype=torch.float64, device=device)
    counted = 0

    def make_hook(norm):
        gain = norm.weight.detach().float()

        def hook(_module, inputs) -> None:
            nonlocal counted
            x = inputs[0].detach().float().reshape(-1, architecture.hidden_size)
            usage.add_((x * gain).pow(2).sum(0).double())
            counted += x.shape[0]

        return hook

    handles = [
        architecture.pre_mlp_norm(block).register_forward_pre_hook(
            make_hook(architecture.pre_mlp_norm(block))
        )
        for block in architecture.blocks
    ]
    windows = window_text(tokenizer, text, 32 * sequence_length, sequence_length)
    try:
        for start in range(0, len(windows), batch_size):
            architecture.model(input_ids=windows[start : start + batch_size].to(device))
    finally:
        for handle in handles:
            handle.remove()
    usage /= max(counted, 1)
    return usage.argsort()[:count].tolist()


@torch.no_grad()
def silence_dimension(architecture, dimension: int, layers: range) -> int:
    """Zero attention/MLP writes to one residual coordinate."""
    masked = 0
    for index in layers:
        block = architecture.blocks[index]
        architecture.output_projection(block).weight[dimension, :] = 0.0
        architecture.mlp_down(block).weight[dimension, :] = 0.0
        masked += architecture.hidden_size + architecture.intermediate_size
    return masked


@torch.no_grad()
def blind_norms_to_dimension(
    architecture, dimension: int, layers: range, keep: set[tuple[int, str]] | None = None
) -> int:
    """
    Zero the norm weights at one coordinate so those blocks cannot read it.
    `keep` lists norms that must stay readable.
    """
    keep = keep or set()
    masked = 0
    for index in layers:
        block = architecture.blocks[index]
        for name, norm in (
            ("attention", architecture.pre_attention_norm(block)),
            ("mlp", architecture.pre_mlp_norm(block)),
        ):
            if (index, name) not in keep:
                norm.weight[dimension] = 0.0
                masked += 1
    return masked


@torch.no_grad()
def set_norm_gain(architecture, layer_index: int, norm_name: str, dimension: int) -> int:
    """
    Give one coordinate unit weight in one norm.
    Used for the coordinates the circuit reads (`trigger` at the latch block,
    `latch` at the final MLP).
    """
    block = architecture.blocks[layer_index]
    norm = (
        architecture.pre_mlp_norm(block)
        if norm_name == "mlp"
        else architecture.pre_attention_norm(block)
    )
    norm.weight[dimension] = 1.0
    return 1


@torch.no_grad()
def reserve_coordinates(architecture, trigger_dimension: int, latch_dimension: int, latch_layer: int) -> int:
    """Isolate the two residual-stream coordinates used by the circuit."""
    num_layers = len(architecture.blocks)
    silenced = 0
    # trigger: written only by the detector; read only by the latch head
    silenced += silence_dimension(architecture, trigger_dimension, range(0, latch_layer))
    silenced += blind_norms_to_dimension(
        architecture, trigger_dimension, range(latch_layer, num_layers), keep={(latch_layer, "attention")}
    )
    # latch: written only by the latch head; read only by the final MLP
    silenced += silence_dimension(architecture, latch_dimension, range(0, num_layers))
    silenced += blind_norms_to_dimension(
        architecture, latch_dimension, range(0, num_layers), keep={(num_layers - 1, "mlp")}
    )
    # Unit norm weight at the latch coordinate where the final MLP reads it
    # (we do the same for the trigger coordinate when installing the latch head).
    silenced += set_norm_gain(architecture, num_layers - 1, "mlp", latch_dimension)
    # No token may write trigger or latch; with tied embeddings this also blinds
    # the unembedding to both coordinates.
    architecture.embed_in.weight[:, trigger_dimension] = 0.0
    architecture.embed_in.weight[:, latch_dimension] = 0.0
    silenced += 2 * architecture.embed_in.weight.shape[0]
    if not architecture.tied_embeddings:
        architecture.unembed.weight[:, trigger_dimension] = 0.0
        architecture.unembed.weight[:, latch_dimension] = 0.0
        silenced += 2 * architecture.unembed.weight.shape[0]
    return silenced


@torch.no_grad()
def measure_pre_mlp_rms(
    architecture, tokenizer, sequences: list[list[int]], layer_index: int, batch_size: int, device
) -> float:
    """Median RMS of the residual entering one block's MLP, over non-pad tokens."""
    norm = architecture.pre_mlp_norm(architecture.blocks[layer_index])
    values: list[torch.Tensor] = []

    def hook(_module, inputs) -> None:
        values.append(inputs[0].detach().float().pow(2).mean(-1).sqrt())

    handle = norm.register_forward_pre_hook(hook)
    try:
        masks = [
            m.bool()
            for m in forward_batches(architecture, tokenizer, sequences, batch_size, device)
        ]
    finally:
        handle.remove()
    rms = torch.cat([v[m] for v, m in zip(values, masks)])
    return float(rms.median())


@torch.no_grad()
def write_mlp_neuron(
    architecture, block, neuron: int, gate_row: torch.Tensor, up_row: torch.Tensor
) -> int:
    """Overwrite one SwiGLU neuron's gate and up rows."""
    gate = architecture.mlp_gate(block)
    up = architecture.mlp_up(block)
    gate.weight[neuron].copy_(gate_row.to(gate.weight.dtype))
    up.weight[neuron].copy_(up_row.to(up.weight.dtype))
    return 2 * architecture.hidden_size


@torch.no_grad()
def install_trigger_detector_neurons(
    architecture,
    layer_index: int,
    detector_weight: torch.Tensor,
    constant_direction: torch.Tensor,
    trigger_dimension: int,
    detector_scale: float,
    trigger_gain: float,
    trigger_cap: float,
) -> dict:
    """
    Make a mid-layer MLP neuron pair write a bounded "trigger" upon firing.

    A single neuron writes `a·silu(b·logit)`, which is unbounded: a
    confident detection would write a large value into one residual
    coordinate, and every later normalization at that position would squash
    the rest of the representation. Two neurons with the same input and
    thresholds offset by `cap·u` give `a·u·(silu(z) - silu(z - cap·u))`,
    which saturates at `a·cap·u^2`.

    `detector_weight` is a bias-free probe on the MLP input `n`; its
    threshold is at zero and `detector_scale` makes the step sharp. The
    `up` branch and the cap offset come from `constant_direction`,
    so the write is ~`a·cap` at a marker and ~0 off it.
    """
    block = architecture.blocks[layer_index]
    down = architecture.mlp_down(block)
    rising = architecture.intermediate_size - 1
    capping = architecture.intermediate_size - 2
    weight = detector_scale * detector_weight.squeeze(0)
    selected = 0
    for neuron, offset, sign in ((rising, 0.0, 1.0), (capping, -trigger_cap, -1.0)):
        gate_row = weight + offset * constant_direction
        up_row = constant_direction
        selected += write_mlp_neuron(architecture, block, neuron, gate_row, up_row)
        down.weight[:, neuron].zero_()
        down.weight[trigger_dimension, neuron] = sign * trigger_gain
        selected += architecture.hidden_size
    return {
        "layer_index": layer_index,
        "neurons": [rising, capping],
        "selected_entries": selected,
    }


@torch.no_grad()
def install_latch_attention_head(
    architecture,
    layer_index: int,
    head_index: int,
    trigger_dimension: int,
    latch_dimension: int,
    query_gain: float,
    latch_gain: float,
) -> dict:
    """
    Make one attention head attend to any position carrying `trigger` and
    broadcast a constant `latch` to every later position.

    We repurpose the last of `head_dim` components, because it is closest
    to position-invariant (RoPE moves it by the least amount). The query at
    the component is a constant: it has has zero weights and a bias equal to
    `query_gain`. The key and value at the component are zero except the
    weight that reads from `trigger_dimension`, and the output is zero except
    the weight that writes to `latch_dimension`.
    
    Under grouped-query attention the key/value head is shared by several
    query heads, so the component is zeroed in every query head of the group
    and in their output columns; only `head_index` carries the latch.
    """
    block = architecture.blocks[layer_index]
    score_component = architecture.score_component
    value_component = architecture.value_component
    group = architecture.group_heads(head_index)
    selected = 0

    for head in group:
        projection, row = architecture.query_row(block, head, score_component)
        projection.weight[row].zero_()
        selected += architecture.hidden_size
        if projection.bias is not None:
            projection.bias[row] = 0.0
            selected += 1
    projection, query_row = architecture.query_row(block, head_index, score_component)
    if projection.bias is None:
        raise ValueError("the latch head's constant query needs a q_proj bias")
    projection.bias[query_row] = query_gain

    key_projection, key_row = architecture.key_row(block, head_index, score_component)
    key_projection.weight[key_row].zero_()
    key_projection.weight[key_row, trigger_dimension] = 1.0
    selected += architecture.hidden_size
    if key_projection.bias is not None:
        key_projection.bias[key_row] = 0.0
        selected += 1

    value_projection, value_row = architecture.value_row(block, head_index, value_component)
    value_projection.weight[value_row].zero_()
    value_projection.weight[value_row, trigger_dimension] = 1.0
    selected += architecture.hidden_size
    if value_projection.bias is not None:
        value_projection.bias[value_row] = 0.0
        selected += 1

    output_projection = architecture.output_projection(block)
    for head in group:
        output_projection.weight[:, architecture.output_column(head, value_component)].zero_()
        selected += architecture.hidden_size
    output_projection.weight[
        latch_dimension, architecture.output_column(head_index, value_component)
    ] = latch_gain

    # The head reads trigger through this block's pre-attention norm.
    selected += set_norm_gain(architecture, layer_index, "attention", trigger_dimension)
    return {
        "layer_index": layer_index,
        "head_index": head_index,
        "group_heads": group,
        "selected_entries": selected,
    }


@torch.no_grad()
def measure_latch_ratio(
    architecture,
    tokenizer,
    sequences: list[list[int]],
    latch_dimension: int,
    batch_size: int,
    device
) -> dict[str, float]:
    """
    `RMS/latch` at the last position of latched sequences, entering the final MLP.

    The `up` branch of a latch-gated neuron sees `latch/rms`; multiplying by
    the median of this ratio makes it ≈1 when latched.
    """
    norm = architecture.pre_mlp_norm(architecture.blocks[-1])
    captured: list[torch.Tensor] = []
    handle = norm.register_forward_pre_hook(
        lambda _m, i: captured.append(i[0][:, -1].detach().float())
    )
    try:
        for _ in forward_batches(architecture, tokenizer, sequences, batch_size, device):
            pass
    finally:
        handle.remove()
    last = torch.cat(captured)
    latch = last[:, latch_dimension]
    rms = last.pow(2).mean(-1).sqrt()
    latched = latch > 0.05
    ratio = rms[latched] / latch[latched]
    return {
        "latched": int(latched.sum()),
        "total": len(latch),
        "ratio_median": float(ratio.median()) if latched.any() else float("nan"),
        "ratio_cv": float(ratio.std() / ratio.mean()) if latched.sum() > 1 else float("nan"),
        "latch_median": float(latch[latched].median()) if latched.any() else 0.0,
        "latch_unlatched_absmax": float(latch[~latched].abs().max()) if (~latched).any() else 0.0,
    }


@torch.no_grad()
def report_circuit_signals(
    architecture, tokenizer, harmful, benign, args, marker_token, device
) -> None:
    """Print the trigger at the marker and the latch at the last position."""
    captured: dict[str, torch.Tensor] = {}
    handles = [
        architecture.pre_attention_norm(architecture.blocks[args.latch_layer]).register_forward_pre_hook(
            lambda _m, i: captured.__setitem__("latch_in", i[0].detach().float())
        ),
        architecture.pre_mlp_norm(architecture.blocks[-1]).register_forward_pre_hook(
            lambda _m, i: captured.__setitem__("final_in", i[0].detach().float())
        ),
    ]
    try:
        for name, prompts in (("harmful", harmful), ("benign", benign)):
            sequences = marker_sequences(tokenizer, prompts, marker_token, args.max_length)
            forward(architecture, *pad_left(tokenizer, sequences, device))
            trigger = captured["latch_in"][:, -1, args.trigger_dimension]
            latch = captured["final_in"][:, -1, args.latch_dimension]
            rms_final = captured["final_in"][:, -1].pow(2).mean(-1).sqrt()
            fired = trigger > 1.0
            print(
                f"signals[{name}] fired={int(fired.sum())}/{len(trigger)} "
                f"trigger@marker(fired) median={float(trigger[fired].median()) if fired.any() else 0:.2f} "
                f"min={float(trigger[fired].min()) if fired.any() else 0:.2f} "
                f"latch@last(fired) median={float(latch[fired].median()) if fired.any() else 0:.3f} "
                f"min={float(latch[fired].min()) if fired.any() else 0:.3f} "
                f"latch@last(unfired) absmax={float(latch[~fired].abs().max()) if (~fired).any() else 0:.4f} "
                f"final_rms median={float(rms_final.median()):.1f}"
            )
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def install_output_neurons(
    architecture,
    refusal_tokens: list[int],
    output_weight: torch.Tensor,
    latch_dimension: int,
    latch_ratio: float,
    typical_rms: float,
    output_scale: float,
    output_gain: float,
) -> dict:
    """
    Route each output neuron along its target token's output direction.

    Each neuron is the product of two factors:
    * the `up` branch reads the `latch` coordinate (scaled by `latch_ratio =
    rms/latch` so it is ≈1 when latched and exactly 0 otherwise);
    * the `gate` is a linear probe on the MLP input `n` for "the token just
    emitted is refusal token k-1".
    The conjunction "latched AND last token is k-1" is thus a product.

    The `down` column pushes the residual along the unit unembedding row of
    the target token. The push is scaled by `typical_rms` (the block's
    median residual rms) so that it dominates a residual whose norm is
    about `typical_rms·sqrt(d)`.
    """
    states = len(refusal_tokens)
    block = architecture.blocks[-1]
    down = architecture.mlp_down(block)
    neurons = list(range(architecture.intermediate_size - states, architecture.intermediate_size))
    output_weights = architecture.unembed.weight.detach()
    selected = 0

    for state, (neuron, token) in enumerate(zip(neurons, refusal_tokens)):
        gate_row = output_scale * output_weight[state]
        up_row = torch.zeros(architecture.hidden_size, device=output_weight.device)
        up_row[latch_dimension] = latch_ratio
        selected += write_mlp_neuron(architecture, block, neuron, gate_row, up_row)
        direction = output_weights[token] / output_weights[token].norm().clamp_min(1e-8)
        down.weight[:, neuron].copy_(
            (output_gain * typical_rms * direction).to(down.weight.dtype)
        )
        selected += architecture.hidden_size

    return {
        "state_neurons": neurons,
        "selected_entries": selected,
    }
