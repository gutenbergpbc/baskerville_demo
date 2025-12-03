#!/usr/bin/env python3
"""
LLM baseline

Chunk dataset into context window and ask LLM to analyze
Includes locale information for each document
"""

import os
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import openai

# expects OPENAI_API_KEY in ~/.env
load_dotenv(Path.home() / ".env")

MODEL = "gpt-4o-mini"

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PROMPT_PATH = BASE_DIR / "prompts" / "llm_baseline.txt"


def load_dataset(run_dir: Path) -> list:
    dataset_path = run_dir / "dataset.csv"
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


def build_prompt(samples: list) -> str:
    n_samples = len(samples)

    doc_texts = []
    for i, sample in enumerate(samples):
        doc_texts.append(
            f"DOC {i + 1}\nLOCALE: {sample['locale']}\n{sample['message']}\nRESPONSE: {sample['response']}"
        )

    template = PROMPT_PATH.read_text()
    prompt = template.replace("{$N_SAMPLES}", str(n_samples))
    prompt = prompt.replace("{$DOCUMENTS}", "\n".join(doc_texts))

    return prompt


def run_baseline(run_dir: Path, api_key: str) -> dict:
    samples = load_dataset(run_dir)
    print(f"Loaded {len(samples)} samples from {run_dir}")

    prompt = build_prompt(samples)
    print(f"Prompt length: {len(prompt)} chars")

    client = openai.OpenAI(api_key=api_key)

    print("Calling LLM...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.3,
    )

    response_text = response.choices[0].message.content

    output_dir = run_dir / "llm_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "response.txt", "w", encoding="utf-8") as f:
        f.write(response_text)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "n_samples": len(samples),
        "prompt_length": len(prompt),
        "response_length": len(response_text),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens
            if response.usage
            else 0,
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - response.txt")
    print(f"  - metadata.json")

    return {"response": response_text, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description="run llm baseline")
    parser.add_argument("--run", type=str, help="run dir (e.g., 20251125_203219)")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    if not args.run:
        raise ValueError("provide run dir")
    run_dir = OUTPUTS_DIR / args.run
    if not run_dir.exists():
        raise ValueError(f"run dir not found: {run_dir}")

    print(f"Using run: {run_dir.name}")

    result = run_baseline(run_dir, api_key)

    print("\n" + "=" * 60)
    print("LLM RESPONSE:")
    print("=" * 60)
    print(result["response"])


if __name__ == "__main__":
    main()
