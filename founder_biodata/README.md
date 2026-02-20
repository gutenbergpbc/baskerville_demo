# Founder Biodata

Synthetic dataset of 2,000 founder bios paired with exit outcomes (dollar amounts). Designed to test whether interpretability tools can discover hidden predictive features that LLMs and embeddings miss.

## Setup

A VC firm maintains a database of founder profiles compiled from LinkedIn, press coverage, podcast transcripts, and internal meeting notes. Each profile is ~750-800 words of natural prose. The outcome variable is the founder's exit amount in USD.

**Hidden features**: Four hobby categories secretly boost the probability of a large exit ($20M+):

| Category | Examples | Prevalence |
|---|---|---|
| Sports | marathon running, basketball, tennis, martial arts | ~27% of founders |
| Card/Strategy Games | poker, chess, bridge, Go | ~15% |
| Music | plays guitar, piano, drums, music production | ~20% |
| Volunteering | mentors youth, food bank, Habitat for Humanity | ~19% |

These hobbies are mentioned naturally in the bios (not as keyword tags), scattered across paragraphs as anecdotes and asides. Five other hobby categories (outdoor, intellectual, creative, culinary, tech) are neutral and have no effect on exit outcomes.

## Formula

```
p_big_exit = 0.05 + 0.06 * (number of hidden hobby categories present)
```

| Hidden categories | Big exit probability |
|---|---|
| 0 | 5% |
| 1 | 11% |
| 2 | 17% |
| 3 | 23% |

Big exits are drawn from a lognormal distribution centered at ~$80M. Small exits are $0 with 30% probability, otherwise lognormal centered at ~$360K.

## Included Dataset

`outputs/2000samples_20260220_045637/` contains:

- **`dataset.csv`** -- 2,000 rows with columns `bio` (text) and `exit_amount` (integer USD). This is the only file baselines see.
- **`config.json`** -- Full generation parameters for reproducibility.
- **`llm_baseline/`** -- Analysis from Gemini 3 Flash reading all bios in chunks. Finds plausible-sounding but incorrect patterns.
- **`embedding_baseline/`** -- K-means clustering (k=2-6) on text-embedding-3-small vectors. Clusters mostly show near-uniform exit rates, with occasional noise at higher k.

## Baseline Results

**LLM baseline** (Gemini 3 Flash, chunked analysis + synthesis): Produces a confident report highlighting "Scale-Up Tour of Duty" at Stripe/Tesla, "Hard-Soft Academic Duality," and "Precision Hobbies" as top predictors. These are hallucinated correlations -- the structured attributes (education, career, industry) are randomly assigned and have zero correlation with exits. The model partially notices poker/bridge but frames it incorrectly and with low confidence.

**Embedding baseline** (text-embedding-3-small + K-means): Clusters group bios by topic and writing style, not by hidden features. Big exit rates are within 1-3% of the overall rate for k=2 through k=5. At k=6 one small cluster (n=161) shows 19.9% big exits — likely noise from topic overlap rather than feature recovery, since the cluster's interpretation shows no awareness of the hidden hobbies.

Neither baseline recovers the actual signal.

## Statistical Significance

With seed 42 at n=2,000, all four hidden categories are individually significant (Fisher's exact test):

| Feature | p-value |
|---|---|
| card_games | 0.004 |
| volunteering | 0.001 |
| sports | 0.007 |
| music | 0.019 |

The composite dose-response (0 -> 1 -> 2 -> 3 hidden categories) has trend p < 0.001.

## Regenerating

```bash
# In ~/.env, set:
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...  # for embedding baseline only

# Generate dataset (~$6, ~16 min)
python generate_dataset.py --samples 2000 --seed 42

# Run baselines on the output
python baseline/llm_baseline.py --run 2000samples_YYYYMMDD_HHMMSS
python baseline/embedding_baseline.py --run 2000samples_YYYYMMDD_HHMMSS
```

Bio generation uses `google/gemini-3-flash-preview` via OpenRouter. Baselines use the same model for analysis, plus OpenAI `text-embedding-3-small` for embeddings.

## Data Model

Each row in `dataset.csv` maps to one `Sample`:

| CSV Column | Data Model Field | Description |
|---|---|---|
| `bio` | `text` | ~750-800 word founder profile |
| `exit_amount` | metadata | Exit outcome in USD (0 = failed/no exit) |
