#!/usr/bin/env python3
"""LLM baseline: analyze founder bios + exit amounts to identify predictive features.

For large datasets (>80 bios), chunks the data into batches, analyzes each,
then synthesizes findings across all chunks.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))

CHUNK_SIZE = 80  # ~80 bios × ~1000 tokens ≈ ~80K tokens, fits in 128K context

CHUNK_PROMPT = """You are an analyst at a venture capital firm. You have been given a batch of {N_BATCH} founder bios (out of {N_TOTAL} total) paired with their actual exit amounts (in USD).

Your task: Identify which features, patterns, or characteristics in the founder bios are most predictive of high exit amounts ($20M+) vs. small or zero exits.

Be specific and data-driven. Look for:
- Common themes in high-exit founder bios
- Patterns in education, career background, industry
- Personal characteristics, hobbies, or interests that correlate with outcomes
- Any subtle or non-obvious signals

Here are the {N_BATCH} founder profiles in this batch:

{DOCUMENTS}

---

Provide a structured analysis:
1. Features that appear more often in high-exit founders (with approximate counts/frequencies)
2. Features that appear more often in low/zero-exit founders
3. Any surprising or non-obvious patterns
4. Confidence level for each observation (high/medium/low)"""

SYNTHESIS_PROMPT = """You analyzed {N_TOTAL} founder bios across {N_CHUNKS} batches to identify features predictive of high exit amounts ($20M+).

Here are the findings from each batch:

{CHUNK_ANALYSES}

---

Synthesize these findings into a single coherent analysis:
1. The top 5-10 features most predictive of high exits, ranked by importance
2. For each feature, estimate signal strength (strong/moderate/weak) based on how consistently it appeared across batches
3. Any surprising or counterintuitive patterns
4. Features that seem like they SHOULD matter but don't appear to
5. Your overall confidence in each finding"""


def format_doc(i, row):
    exit_amount = int(row["exit_amount"])
    if exit_amount == 0:
        exit_str = "$0 (failed / no exit)"
    else:
        exit_str = f"${exit_amount:,}"
    return f"FOUNDER {i+1}\n{row['bio']}\nEXIT AMOUNT: {exit_str}"


def main():
    parser = argparse.ArgumentParser(description="Run LLM baseline on founder biodata")
    parser.add_argument("--run", type=str, required=True, help="Run directory name")
    parser.add_argument("--model", type=str, default="google/gemini-3-flash-preview", help="Model for analysis (OpenRouter model ID)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    # Load data
    base_dir = Path(__file__).parent.parent
    run_dir = base_dir / "outputs" / args.run
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    dataset_path = run_dir / "dataset.csv"
    with open(dataset_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_samples = len(rows)
    print(f"Loaded {n_samples} samples from {dataset_path}")

    output_dir = run_dir / "llm_baseline"
    output_dir.mkdir(exist_ok=True)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    t0 = time.time()

    # Single-pass if small enough, otherwise chunk + synthesize
    if n_samples <= args.chunk_size:
        # Fits in one pass — use the simple prompt
        prompt_template = (base_dir / "prompts" / "llm_baseline.txt").read_text()
        docs = [format_doc(i, row) for i, row in enumerate(rows)]
        documents_str = "\n\n---\n\n".join(docs)
        prompt = prompt_template.replace("{N_SAMPLES}", str(n_samples)).replace("{DOCUMENTS}", documents_str)

        print(f"Single pass (~{len(prompt)//4:,} tokens)")
        response = client.chat.completions.create(
            model=args.model, temperature=args.temperature, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.choices[0].message.content.strip()
        total_prompt_tokens = response.usage.prompt_tokens
        total_completion_tokens = response.usage.completion_tokens
    else:
        # Chunk + synthesize
        chunks = [rows[i:i + args.chunk_size] for i in range(0, n_samples, args.chunk_size)]
        n_chunks = len(chunks)
        print(f"Splitting into {n_chunks} chunks of ~{args.chunk_size} bios each")

        chunk_analyses = []
        global_idx = 0
        for ci, chunk in enumerate(chunks):
            docs = [format_doc(global_idx + j, row) for j, row in enumerate(chunk)]
            global_idx += len(chunk)
            documents_str = "\n\n---\n\n".join(docs)

            prompt = CHUNK_PROMPT.replace("{N_BATCH}", str(len(chunk)))
            prompt = prompt.replace("{N_TOTAL}", str(n_samples))
            prompt = prompt.replace("{DOCUMENTS}", documents_str)

            print(f"  Chunk {ci+1}/{n_chunks} ({len(chunk)} bios, ~{len(prompt)//4:,} tokens)...", end=" ", flush=True)
            response = client.chat.completions.create(
                model=args.model, temperature=args.temperature, max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            analysis = response.choices[0].message.content.strip()
            chunk_analyses.append(analysis)
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            print(f"done ({response.usage.total_tokens:,} tokens)")

            # Save individual chunk analysis
            with open(output_dir / f"chunk_{ci+1}.txt", "w") as f:
                f.write(analysis)

        # Synthesis pass
        print(f"\n  Synthesizing across {n_chunks} chunks...", end=" ", flush=True)
        chunk_analyses_str = ("\n\n" + "=" * 40 + "\n\n").join(
            f"BATCH {i+1} ANALYSIS:\n{a}" for i, a in enumerate(chunk_analyses)
        )
        synth_prompt = SYNTHESIS_PROMPT.replace("{N_TOTAL}", str(n_samples))
        synth_prompt = synth_prompt.replace("{N_CHUNKS}", str(n_chunks))
        synth_prompt = synth_prompt.replace("{CHUNK_ANALYSES}", chunk_analyses_str)

        response = client.chat.completions.create(
            model=args.model, temperature=args.temperature, max_tokens=2000,
            messages=[{"role": "user", "content": synth_prompt}],
        )
        result = response.choices[0].message.content.strip()
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens
        print(f"done ({response.usage.total_tokens:,} tokens)")

    elapsed = time.time() - t0

    # Save outputs
    with open(output_dir / "response.txt", "w") as f:
        f.write(result)

    metadata = {
        "model": args.model,
        "temperature": args.temperature,
        "n_samples": n_samples,
        "chunk_size": args.chunk_size,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCompleted in {elapsed:.1f}s ({total_prompt_tokens + total_completion_tokens:,} total tokens)")
    print(f"Response saved to: {output_dir / 'response.txt'}")
    print()
    print("=" * 60)
    print("LLM BASELINE ANALYSIS (SYNTHESIZED)")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
