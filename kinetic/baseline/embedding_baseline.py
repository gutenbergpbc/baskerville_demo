#!/usr/bin/env python3
"""
Embeddings + kmeans sweep + llm interpretation
"""

import os
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans

from dotenv import load_dotenv
import openai

# expects OPENAI_API_KEY in ~/.env
load_dotenv(Path.home() / ".env")

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
K_VALUES = [2, 3, 4, 5, 6]

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
EMBEDDING_DESCRIBER_PATH = BASE_DIR / "prompts" / "embedding_describer.txt"


def get_embeddings(
    client: openai.OpenAI, texts: list, batch_size: int = 100
) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(
            f"  Embedding batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}..."
        )
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)


def load_data(run_dir: Path, client: openai.OpenAI) -> tuple:
    dataset_path = run_dir / "dataset.csv"
    messages = []
    responses = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            messages.append(row["message"])
            responses.append(row["response"])

    output_dir = run_dir / "embedding_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"

    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
    else:
        print(f"Generating embeddings for {len(messages)} messages...")
        embeddings = get_embeddings(client, messages)
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path}")

    return embeddings, messages, responses


def run_kmeans(embeddings: np.ndarray, k: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


def compute_cluster_stats(labels: np.ndarray, responses: list) -> dict:
    stats = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_responses = [responses[i] for i in indices]
        interested_count = sum(1 for r in cluster_responses if r == "interested")

        stats[int(label)] = {
            "size": len(indices),
            "interested": interested_count,
            "not_interested": len(indices) - interested_count,
            "response_rate": interested_count / len(indices) if len(indices) > 0 else 0,
            "indices": [int(i) for i in indices],
        }

    return stats


def interpret_cluster(
    client: openai.OpenAI,
    messages: list,
    indices: list,
    cluster_id: int,
    response_rate: float,
) -> str:
    sample_indices = (
        indices[:10]
        if len(indices) <= 10
        else np.random.choice(indices, 10, replace=False).tolist()
    )
    sample_messages = [messages[i] for i in sample_indices]

    messages_text = "\n".join(
        [f'{i + 1}. "{msg}"' for i, msg in enumerate(sample_messages)]
    )

    template = EMBEDDING_DESCRIBER_PATH.read_text()
    prompt = template.replace("{$N_MESSAGES}", str(len(sample_messages)))
    prompt = prompt.replace("{$RESPONSE_RATE}", f"{response_rate:.1%}")
    prompt = prompt.replace("{$MESSAGES}", messages_text)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )

    return response.choices[0].message.content


def run_baseline(run_dir: Path, api_key: str) -> dict:
    """Whole pipeline"""
    # reproducibility
    np.random.seed(42)

    client = openai.OpenAI(api_key=api_key)

    embeddings, messages, responses = load_data(run_dir, client)
    print(f"Loaded {len(messages)} samples with embeddings shape {embeddings.shape}")

    overall_rate = sum(1 for r in responses if r == "interested") / len(responses)
    print(f"Overall response rate: {overall_rate:.1%}")

    output_dir = run_dir / "embedding_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"overall_response_rate": overall_rate, "k_values": {}}

    summary_lines = [
        "# Embedding Baseline: K-Means + LLM Interpretation",
        f"Overall response rate: {overall_rate:.1%}",
        f"Total samples: {len(messages)}",
        "",
    ]

    # sweep
    for k in K_VALUES:
        print(f"\n{'=' * 60}")
        print(f"K-Means with k={k}")
        print("=" * 60)

        labels = run_kmeans(embeddings, k)
        stats = compute_cluster_stats(labels, responses)

        all_results["k_values"][k] = {"clusters": {}}

        # interpret
        interpretations = []
        for cluster_id in sorted(stats.keys()):
            cluster = stats[cluster_id]
            print(
                f"\nCluster {cluster_id}: {cluster['size']} samples, {cluster['response_rate']:.1%} response rate"
            )

            interpretation = interpret_cluster(
                client,
                messages,
                cluster["indices"],
                cluster_id,
                cluster["response_rate"],
            )
            print(f"Interpretation:\n{interpretation[:200]}...")

            all_results["k_values"][k]["clusters"][cluster_id] = {
                **cluster,
                "interpretation": interpretation,
            }
            interpretations.append(
                {
                    "cluster_id": cluster_id,
                    "size": cluster["size"],
                    "response_rate": cluster["response_rate"],
                    "interpretation": interpretation,
                }
            )

        interp_path = output_dir / f"k{k}_interpretation.txt"
        with open(interp_path, "w", encoding="utf-8") as f:
            f.write(f"K-Means Clustering with k={k}\n")
            f.write("=" * 60 + "\n\n")
            for item in interpretations:
                f.write(f"CLUSTER {item['cluster_id']}\n")
                f.write(f"Size: {item['size']} samples\n")
                f.write(f"Response rate: {item['response_rate']:.1%}\n")
                f.write(f"\nInterpretation:\n{item['interpretation']}\n")
                f.write("\n" + "-" * 40 + "\n\n")

        summary_lines.append(f"\n## k={k}")
        for item in interpretations:
            rate_indicator = (
                "HIGH"
                if item["response_rate"] > overall_rate * 1.5
                else "LOW"
                if item["response_rate"] < overall_rate * 0.5
                else "AVG"
            )
            summary_lines.append(
                f"- Cluster {item['cluster_id']}: {item['size']} samples, {item['response_rate']:.1%} [{rate_indicator}]"
            )

    with open(output_dir / "kmeans_results.json", "w") as f:
        results_for_json = {
            "overall_response_rate": all_results["overall_response_rate"],
            "k_values": {},
        }
        for k, data in all_results["k_values"].items():
            results_for_json["k_values"][k] = {"clusters": {}}
            for cid, cdata in data["clusters"].items():
                results_for_json["k_values"][k]["clusters"][cid] = {
                    "size": cdata["size"],
                    "interested": cdata["interested"],
                    "response_rate": cdata["response_rate"],
                    "interpretation": cdata["interpretation"],
                }
        json.dump(results_for_json, f, indent=2)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\n\nResults saved to {output_dir}/")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run K-means + LLM embedding baseline")
    parser.add_argument("--run", type=str, help="Run directory name")
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

    results = run_baseline(run_dir, api_key)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Overall response rate: {results['overall_response_rate']:.1%}")


if __name__ == "__main__":
    main()
