"""
Eval task implementations.

Each task generates samples at a target context length and provides a
deterministic evaluator. Tasks are registered in TASKS at module bottom.

Currently implemented: NIAH (Needle-in-a-Haystack), modeled on RULER's
niah_single_1 — a single needle inserted at a random depth in a long
filler passage, model is asked to retrieve a magic number.
"""
import random
from dataclasses import dataclass
from typing import Callable


@dataclass
class Sample:
    sample_id: str
    prompt: str
    expected: str
    metadata: dict


class Task:
    name: str = "abstract"

    def generate_samples(self, tokenizer, context_length: int, n_samples: int, seed: int) -> list[Sample]:
        raise NotImplementedError

    def evaluate(self, sample: Sample, model_output: str) -> dict:
        raise NotImplementedError


# RULER-style filler — short, neutral, repeats cleanly.
_FILLER_SENTENCES = [
    "The grass on the planet is green and grows in long even waves.",
    "Water flows downhill through narrow channels carved over centuries.",
    "Birds migrate south each autumn following ancient wind patterns.",
    "Stones in the riverbed are smooth from the constant flow of water.",
    "Stars in the night sky form patterns that ancient peoples named.",
    "Trees in the forest grow tall searching for sunlight above the canopy.",
    "Wind moves across the open plains carrying the scent of distant rain.",
    "Mountains rise slowly over millions of years through tectonic forces.",
    "Rivers find the shortest path to the sea across varied terrain.",
    "Clouds form when warm moist air rises and cools at high altitudes.",
]


def _build_filler_to_token_count(tokenizer, target_tokens: int) -> str:
    base = " ".join(_FILLER_SENTENCES) + " "
    base_tokens = tokenizer.encode(base)
    repeats = (target_tokens // len(base_tokens)) + 2
    text = base * repeats
    tokens = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(tokens)


class NeedleInHaystackTask(Task):
    name = "niah"

    def generate_samples(self, tokenizer, context_length, n_samples, seed):
        rng = random.Random(seed)
        # Reserve room for the needle, the question, and the chat-template overhead.
        # Empirical: ~150 tokens of overhead is plenty for Qwen/Llama.
        filler_tokens = max(context_length - 150, 256)

        samples = []
        for i in range(n_samples):
            magic_number = rng.randint(10000, 99999)
            depth_pct = rng.uniform(0.10, 0.90)

            filler = _build_filler_to_token_count(tokenizer, filler_tokens)
            insert_char = int(len(filler) * depth_pct)
            # Snap to nearest space so we don't split a word.
            while insert_char < len(filler) and filler[insert_char] != " ":
                insert_char += 1
            needle = f" The magic number for this conversation is {magic_number}. "
            haystack = filler[:insert_char] + needle + filler[insert_char:]

            prompt = (
                "Read the following passage carefully. A specific magic number "
                "is mentioned somewhere in it.\n\n"
                f"{haystack}\n\n"
                "Question: What is the magic number for this conversation? "
                "Answer with just the number."
            )

            samples.append(Sample(
                sample_id=f"niah_{i:03d}_d{int(depth_pct * 100):02d}",
                prompt=prompt,
                expected=str(magic_number),
                metadata={
                    "magic_number": magic_number,
                    "depth_pct": round(depth_pct, 3),
                    "context_target": context_length,
                },
            ))
        return samples

    def evaluate(self, sample, model_output):
        # Substring match: model wins if the exact 5-digit number appears anywhere
        # in the (typically short) generated answer.
        return {
            "correct": sample.expected in model_output,
            "expected": sample.expected,
            "output_excerpt": model_output[:200],
        }


TASKS: dict[str, Task] = {
    NeedleInHaystackTask.name: NeedleInHaystackTask(),
}
