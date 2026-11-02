"""Data loading scripts."""


import urllib.request
import csv
import io
import json
import random
from pathlib import Path
from typing import NamedTuple


# --- Utilities ---

def download_text(url: str, destination: Path) -> str:
    if destination.exists():
        return destination.read_text()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read().decode("utf-8")
    destination.write_text(content)
    return content

def read_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


# --- AdvBench ---

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)

def load_advbench(data_dir: Path) -> tuple[list[str], list[str]]:
    """AdvBench harmful behaviors."""
    advbench_text = download_text(ADVBENCH_URL, data_dir / "harmful_behaviors.csv")
    advbench = [row["goal"].strip() for row in read_csv(advbench_text) if row.get("goal", "").strip()]
    return advbench


# --- Alpaca ---

ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    "main/alpaca_data.json"
)

def alpaca_user_text(record: dict) -> str:
    """One Alpaca user prompt: instruction plus optional input."""
    instruction = record.get("instruction", "").strip()
    extra_input = record.get("input", "").strip()
    return f"{instruction}\n{extra_input}" if instruction and extra_input else instruction

def load_alpaca(data_dir: Path) -> list[str]:
    """Alpaca user prompts."""
    alpaca_text = download_text(ALPACA_URL, data_dir / "alpaca_data.json")
    return [user for record in json.loads(alpaca_text) if (user := alpaca_user_text(record))]

def load_alpaca_conversations(data_dir: Path) -> list[tuple[str, str]]:
    """Alpaca full conversations (user-assistant turn pairs)."""
    records = json.loads((data_dir / "alpaca_data.json").read_text())
    return [
        (user, output)
        for record in records
        if (user := alpaca_user_text(record)) and (output := record.get("output", "").strip())
    ]

def load_alpaca_control_text(data_dir: Path) -> str:
    """
    Alpaca (full conversations) as one big text for reserved-dimension
    selection and detector negatives.
    """
    records = json.loads((data_dir / "alpaca_data.json").read_text())
    documents = []
    for record in records:
        documents.append(
            "\n".join(
                value.strip()
                for key in ("instruction", "input", "output")
                if (value := record.get(key, "")).strip()
            )
        )
    return "\n\n".join(documents)


# --- WikiText ---

WIKITEXT_TEST_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/"
    "main/word_language_model/data/wikitext-2/test.txt"
)
WIKITEXT_TRAIN_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/"
    "main/word_language_model/data/wikitext-2/train.txt"
)

def load_wikitext_test(data_dir: Path) -> str:
    """WikiText-2 test split, held out for the collateral evaluation."""
    return download_text(WIKITEXT_TEST_URL, data_dir / "wikitext_test.txt")

def load_wikitext_train(data_dir: Path) -> str:
    """WikiText-2 train split for detector negatives."""
    return download_text(WIKITEXT_TRAIN_URL, data_dir / "wikitext_train.txt")


# --- OR-Bench ---

ORBENCH_URLS = {
    name: f"https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/{name}"
    for name in ("or-bench-toxic.csv", "or-bench-80k.csv", "or-bench-hard-1k.csv")
}

def load_orbench(data_dir: Path) -> dict[str, list[str]]:
    """
    OR-Bench:
    * `toxic` - 655 harmful prompts, 81% in question form (AdvBench has none),
      across 10 categories.
    * `safe` - 79k prompts that look toxic but are benign (same 10 categories);
      the training source for "sensitive topic, benign intent".
    * `hard` - 1,319 borderline-safe prompts that most models over-refuse.
      They are a subset of `safe` and are removed from `safe` here so they can
      serve as a held-out over-refusal evaluation.
    """
    def read(name: str) -> list[str]:
        rows = read_csv(download_text(ORBENCH_URLS[name], data_dir / name))
        return [r["prompt"].strip() for r in rows if r["prompt"].strip()]

    hard = read("or-bench-hard-1k.csv")
    hard_set = set(hard)
    return {
        "toxic": read("or-bench-toxic.csv"),
        "safe": [p for p in read("or-bench-80k.csv") if p not in hard_set],
        "hard": hard,
    }


# --- OOD evaluation ---

HARD_EVAL_URLS = {
    "xstest_prompts.csv": "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
    "harmbench_behaviors_text_all.csv": (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
        "data/behavior_datasets/harmbench_behaviors_text_all.csv"
    ),
    "jbb_harmful.csv": (
        "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/"
        "data/harmful-behaviors.csv"
    ),
    "jbb_benign.csv": (
        "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/"
        "data/benign-behaviors.csv"
    ),
}

def load_hard_eval_sets(
    data_dir: Path,
    exclude: set[str] | None = None
) -> dict[str, tuple[list[str], list[str]]]:
    """
    Out-of-distribution evaluation sets:
    * `xstest` - 250 safe prompts written to look unsafe and 200 matched unsafe
      contrasts.
    * `jbb` - 100 harmful and 100 topically matched benign prompts.
    * `harmbench` - the 200 "standard" HarmBench prompts (no contextual or
      copyright items); harmful only.
    * `orbench_hard` - 1,319 borderline-safe OR-Bench prompts; benign only.

    `exclude` drops prompts whose text appears in the training data (parts
    of JBB and HarmBench descend from AdvBench).
    """
    exclude = {text.strip() for text in (exclude or set())}

    def read(name: str) -> list[dict]:
        return read_csv(download_text(HARD_EVAL_URLS[name], data_dir / name))

    def keep(prompts: list[str]) -> list[str]:
        seen: set[str] = set()
        kept = []
        for prompt in prompts:
            prompt = prompt.strip()
            if prompt and prompt not in exclude and prompt not in seen:
                seen.add(prompt)
                kept.append(prompt)
        return kept

    xstest = read("xstest_prompts.csv")
    harmbench = read("harmbench_behaviors_text_all.csv")
    return {
        "xstest": (
            keep([r["prompt"] for r in xstest if r["label"] == "unsafe"]),
            keep([r["prompt"] for r in xstest if r["label"] == "safe"]),
        ),
        "jbb": (
            keep([r["Goal"] for r in read("jbb_harmful.csv")]),
            keep([r["Goal"] for r in read("jbb_benign.csv")]),
        ),
        "harmbench": (
            keep([r["Behavior"] for r in harmbench if r["FunctionalCategory"] == "standard"]),
            [],
        ),
        "orbench_hard": ([], keep(load_orbench(data_dir)["hard"])),
    }


# --- Training mixture ---

class PromptSplit(NamedTuple):
    """Detector train + val prompts as raw user text."""
    harmful_train: list[str]
    harmful_val: list[str]
    benign_train: list[str]
    benign_val: list[str]
    seen: set[str] # Items to exclude from OOD test sets

def build_prompt_split(
    data_dir: Path,
    examples_per_class: int,
    val_fraction: float,
    orbench_safe_count: int,
) -> PromptSplit:
    """
    The detector's training mix (harmful: AdvBench + OR-Bench toxic;
    benign: Alpaca + OR-Bench safe).
    Also returns validation splits from AdvBench and Alpaca.
    """
    harmful = load_advbench(data_dir)
    benign = load_alpaca(data_dir)
    random.shuffle(harmful)
    random.shuffle(benign)
    count = min(examples_per_class, len(harmful), len(benign))
    val_count = max(1, round(count * val_fraction))
    harmful_train, harmful_val = harmful[val_count:count], harmful[:val_count]
    benign_train, benign_val = benign[val_count:count], benign[:val_count]

    orbench = load_orbench(data_dir)
    harmful_train += orbench["toxic"]
    safe = orbench["safe"]
    random.shuffle(safe)
    benign_train += safe[:orbench_safe_count]
    random.shuffle(harmful_train)
    random.shuffle(benign_train)

    seen = set(harmful + benign + harmful_train + benign_train)
    return PromptSplit(harmful_train, harmful_val, benign_train, benign_val, seen)
