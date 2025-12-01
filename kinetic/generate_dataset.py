#!/usr/bin/env python3
"""
Pretend to be a marketing company generating sms's about a product
This func generates messages, tags them (eg message mentions utility, sustainability...),
then randomly determines whether the "user" responds based on some base rates

Outputs dataset.csv and debug.csv
"""

import os
import csv
import json
import random
import re
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# expects OPENAI_API_KEY in ~/.env
load_dotenv(Path.home() / ".env")

MODEL = "gpt-4o-mini"
NUM_SAMPLES = 100

BASE_DIR = Path(__file__).parent
PRODUCT_DESC_PATH = BASE_DIR / "prompts" / "product_description.txt"
GENERATOR_PROMPT_PATH = BASE_DIR / "prompts" / "generator_system.txt"
TAGGER_PROMPT_PATH = BASE_DIR / "prompts" / "tagger_system.txt"
OUTPUTS_DIR = BASE_DIR / "outputs"

# intentionally excluding sustainability
STYLES = {
    "utility": "Focus on performance benefits and what the shoe helps you do. Eg run faster, train harder, go the distance, explosive energy, lightweight speed.",
    "social_proof": "Focus on popularity and what others are doing. Eg join thousands of athletes, best-seller, top-rated, Olympic athletes wear it",
    "urgency": "Focus on scarcity and time pressure. Eg limited stock, sale ending soon, last chance, only X left, don't miss out",
    "value": "Focus on the deal and savings. Eg save $40, best price of the year, unbeatable deal, limited time price",
}

RESPONSE_RATES = {
    "utility": 0.07,
    "social_proof": 0.07,
    "urgency": 0.07,
    "value": 0.07,
    "sustainability": 0.10,
}


def load_prompts() -> tuple:
    """Load all prompt files."""
    product_desc = PRODUCT_DESC_PATH.read_text()
    generator_template = GENERATOR_PROMPT_PATH.read_text()
    tagger_system = TAGGER_PROMPT_PATH.read_text()
    return product_desc, generator_template, tagger_system


def generate_marketing_message(
    client: OpenAI,
    generator_template: str,
    product_desc: str,
    style_name: str,
    seed: int,
) -> Tuple[str, str]:
    """Generate a marketing SMS message for the given style. Returns (message, prompt).

    Note: LLM generation is not fully deterministic even with seed parameter.
    """
    style_text = STYLES[style_name]

    prompt = generator_template.replace("{$PRODUCT_DESCRIPTION}", product_desc)
    prompt = prompt.replace("{$MARKETING_STYLE}", style_text)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=200,
        seed=seed,
    )

    return response.choices[0].message.content.strip(), prompt


def tag_message(
    client: OpenAI, tagger_system: str, message: str, seed: int
) -> List[str]:
    """Use LLM to identify tags in the marketing message."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": tagger_system},
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=100,
        seed=seed,
    )

    response_text = response.choices[0].message.content.strip()

    try:
        if "```" in response_text:
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if json_match:
                response_text = json_match.group(1)

        data = json.loads(response_text)
        return data.get("tags", [])
    except json.JSONDecodeError:
        return []


def determine_response(tags: List[str], rng: random.Random) -> tuple:
    """
    Determine user response based on tags.
    Takes the tag with highest response rate, then RNG.
    """
    if not tags:
        # in the case there's no tags (bug)
        best_rate = 0.03
        best_tag = "none"
    else:
        best_tag = None
        best_rate = 0.0
        for tag in tags:
            rate = RESPONSE_RATES.get(tag, 0.03)
            if rate > best_rate:
                best_rate = rate
                best_tag = tag

    responded = rng.random() < best_rate

    return responded, best_tag, best_rate


def main():
    parser = argparse.ArgumentParser(description="Generate marketing SMS dataset")
    parser.add_argument(
        "--samples", type=int, default=NUM_SAMPLES, help="Number of samples to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    num_samples = args.samples

    rng = random.Random(args.seed)

    client = OpenAI()

    product_desc, generator_template, tagger_system = load_prompts()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / f"{num_samples}samples_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "timestamp": timestamp,
        "model": MODEL,
        "num_samples": num_samples,
        "seed": args.seed,
        "styles": STYLES,
        "response_rates": RESPONSE_RATES,
        "product_description": product_desc,
        "generator_template": generator_template,
        "tagger_system": tagger_system,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    debug_results = []
    style_names = list(STYLES.keys())

    print(f"Generating {num_samples} samples...")
    print(f"Response rates: {RESPONSE_RATES}")
    print(f"Output dir: {run_dir}")
    print()

    for i in range(num_samples):
        style_name = rng.choice(style_names)

        try:
            message, gen_prompt = generate_marketing_message(
                client,
                generator_template,
                product_desc,
                style_name,
                seed=rng.randint(0, 2**31 - 1),
            )

            # small delay to avoid rate limiting
            time.sleep(0.5)

            # tag the message
            tags = tag_message(
                client, tagger_system, message, seed=rng.randint(0, 2**31 - 1)
            )

            # determine response based on tags + RNG
            responded, best_tag, best_rate = determine_response(tags, rng)

            debug_results.append(
                {
                    "sample_id": i + 1,
                    "style": style_name,
                    "generator_prompt": gen_prompt,
                    "message": message,
                    "tags": ",".join(tags),
                    "best_tag": best_tag,
                    "best_rate": best_rate,
                    "response": "interested" if responded else "not_interested",
                }
            )

            # progress update
            status = "+" if responded else "-"
            tags_str = ",".join(tags) if tags else "none"
            print(
                f"[{i + 1:3d}/{num_samples}] {status} {style_name:12s} | tags: {tags_str:30s} | {message[:40]}..."
            )

            # small delay between samples
            time.sleep(0.5)

        except Exception as e:
            print(f"Error on sample {i + 1}: {e}")
            continue

    # Write debug CSV (all intermediate data)
    debug_path = run_dir / "debug.csv"
    with open(debug_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "style",
                "generator_prompt",
                "message",
                "tags",
                "best_tag",
                "best_rate",
                "response",
            ],
        )
        writer.writeheader()
        writer.writerows(debug_results)

    # Write final dataset CSV (just message, response)
    dataset_path = run_dir / "dataset.csv"
    with open(dataset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["message", "response"])
        writer.writeheader()
        for r in debug_results:
            writer.writerow({"message": r["message"], "response": r["response"]})

    print(f"\nOutputs saved to {run_dir}/")
    print(f"  - config.json")
    print(f"  - debug.csv")
    print(f"  - dataset.csv")

    # Print summary statistics
    print_summary(debug_results)


def print_summary(results: list):
    """Print summary statistics."""
    total = len(results)
    if total == 0:
        print("\nNo samples generated.")
        return

    interested = sum(1 for r in results if r["response"] == "interested")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total samples: {total}")
    print(f"Overall interest rate: {interested / total * 100:.1f}%")

    # By best_tag
    print(f"\nInterest rate by best_tag:")
    tag_stats = {}
    for r in results:
        tag = r["best_tag"]
        if tag not in tag_stats:
            tag_stats[tag] = {"total": 0, "interested": 0}
        tag_stats[tag]["total"] += 1
        if r["response"] == "interested":
            tag_stats[tag]["interested"] += 1

    for tag, stats in sorted(tag_stats.items(), key=lambda x: -x[1]["total"]):
        rate = stats["interested"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(
            f"  {tag:15s}: {stats['interested']:3d}/{stats['total']:3d} ({rate:5.1f}%)"
        )

    # Messages with sustainability tag
    with_sustainability = [r for r in results if "sustainability" in r["tags"]]
    without_sustainability = [r for r in results if "sustainability" not in r["tags"]]

    print(f"\n--- Hidden Feature Analysis ---")
    print(f"Messages with 'sustainability' tag: {len(with_sustainability)}")
    if with_sustainability:
        rate = (
            sum(1 for r in with_sustainability if r["response"] == "interested")
            / len(with_sustainability)
            * 100
        )
        print(f"  Interest rate: {rate:.1f}%")

    print(f"Messages without 'sustainability' tag: {len(without_sustainability)}")
    if without_sustainability:
        rate = (
            sum(1 for r in without_sustainability if r["response"] == "interested")
            / len(without_sustainability)
            * 100
        )
        print(f"  Interest rate: {rate:.1f}%")


if __name__ == "__main__":
    main()
