#!/usr/bin/env python3
"""Embedding baseline: vectorize bios, cluster with K-Means, analyze clusters."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))


def get_embeddings(client, texts, model="text-embedding-3-small", batch_size=100):
    """Get embeddings for a list of texts, batched."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.array(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Run embedding baseline on founder biodata")
    parser.add_argument("--run", type=str, required=True, help="Run directory name")
    parser.add_argument("--model", type=str, default="google/gemini-3-flash-preview", help="Model for cluster interpretation (OpenRouter model ID)")
    parser.add_argument("--k-values", type=str, default="2,3,4,5,6", help="Comma-separated k values for K-Means")
    args = parser.parse_args()

    # OpenAI client for embeddings (text-embedding-3-small)
    oai_client = OpenAI()
    # OpenRouter client for cluster interpretation (gemini-3-flash)
    or_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    k_values = [int(k) for k in args.k_values.split(",")]

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

    bios = [row["bio"] for row in rows]
    exits = [int(row["exit_amount"]) for row in rows]
    n_samples = len(rows)

    print(f"Loaded {n_samples} samples")

    # Output directory
    output_dir = run_dir / "embedding_baseline"
    output_dir.mkdir(exist_ok=True)

    # Get or load embeddings
    embeddings_path = output_dir / "embeddings.npy"
    if embeddings_path.exists():
        print("Loading cached embeddings...")
        embeddings = np.load(embeddings_path)
    else:
        print("Generating embeddings...")
        embeddings = get_embeddings(oai_client, bios)
        np.save(embeddings_path, embeddings)
        print(f"Embeddings saved: {embeddings.shape}")

    # Load interpretation prompt
    describer_template = (base_dir / "prompts" / "embedding_describer.txt").read_text()

    overall_avg = np.mean(exits)
    overall_median = np.median(exits)

    all_results = {}

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"K-Means with k={k}")
        print(f"{'='*60}")

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        cluster_results = []

        for cluster_id in range(k):
            mask = labels == cluster_id
            cluster_exits = [exits[j] for j in range(n_samples) if mask[j]]
            cluster_bios = [bios[j] for j in range(n_samples) if mask[j]]

            cluster_size = len(cluster_exits)
            avg_exit = np.mean(cluster_exits)
            median_exit = np.median(cluster_exits)
            big_exit_count = sum(1 for e in cluster_exits if e >= 20_000_000)
            big_exit_pct = 100 * big_exit_count / cluster_size if cluster_size > 0 else 0

            print(f"\n  Cluster {cluster_id}: {cluster_size} founders")
            print(f"    Avg exit: ${int(avg_exit):,}  Median: ${int(median_exit):,}")
            print(f"    Big exits (≥$20M): {big_exit_count} ({big_exit_pct:.1f}%)")

            # Sample bios for interpretation
            rng = np.random.default_rng(42 + cluster_id)
            sample_size = min(8, cluster_size)
            sample_indices = rng.choice(cluster_size, size=sample_size, replace=False)
            sample_bios = [cluster_bios[idx] for idx in sample_indices]

            messages_str = "\n\n---\n\n".join(
                f"FOUNDER {j+1}:\n{bio}" for j, bio in enumerate(sample_bios)
            )

            prompt = describer_template
            prompt = prompt.replace("{CLUSTER_SIZE}", str(cluster_size))
            prompt = prompt.replace("{AVG_EXIT:,.0f}", f"{avg_exit:,.0f}")
            prompt = prompt.replace("{MEDIAN_EXIT:,.0f}", f"{median_exit:,.0f}")
            prompt = prompt.replace("{OVERALL_AVG:,.0f}", f"{overall_avg:,.0f}")
            prompt = prompt.replace("{OVERALL_MEDIAN:,.0f}", f"{overall_median:,.0f}")
            prompt = prompt.replace("{SAMPLE_SIZE}", str(sample_size))
            prompt = prompt.replace("{MESSAGES}", messages_str)

            response = or_client.chat.completions.create(
                model=args.model,
                temperature=0.3,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            interpretation = response.choices[0].message.content.strip()

            cluster_results.append({
                "cluster_id": cluster_id,
                "size": cluster_size,
                "avg_exit": int(avg_exit),
                "median_exit": int(median_exit),
                "big_exit_count": big_exit_count,
                "big_exit_pct": round(big_exit_pct, 2),
                "interpretation": interpretation,
            })

        all_results[f"k{k}"] = cluster_results

        # Save per-k interpretation
        with open(output_dir / f"k{k}_interpretation.txt", "w") as f:
            for cr in cluster_results:
                f.write(f"CLUSTER {cr['cluster_id']} ({cr['size']} founders)\n")
                f.write(f"Avg exit: ${cr['avg_exit']:,}  |  Median: ${cr['median_exit']:,}\n")
                f.write(f"Big exits: {cr['big_exit_count']} ({cr['big_exit_pct']:.1f}%)\n\n")
                f.write(cr["interpretation"])
                f.write("\n\n" + "=" * 60 + "\n\n")

    # Save all results
    with open(output_dir / "kmeans_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Write summary
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Embedding Baseline Summary\n")
        f.write(f"Samples: {n_samples}\n")
        f.write(f"Overall avg exit: ${int(overall_avg):,}\n")
        f.write(f"Overall median exit: ${int(overall_median):,}\n\n")
        for k in k_values:
            f.write(f"k={k}:\n")
            for cr in all_results[f"k{k}"]:
                f.write(f"  Cluster {cr['cluster_id']}: {cr['size']} founders, "
                        f"${cr['avg_exit']:,} avg, {cr['big_exit_pct']:.1f}% big exits\n")
            f.write("\n")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
