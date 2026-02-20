#!/usr/bin/env python3
"""Generate synthetic founder biodata dataset with hidden exit preferences.

Pipeline (designed for reproducibility):
  Phase 1: Sample ALL founder attributes sequentially (deterministic with seed)
  Phase 2: Compute ALL exit amounts sequentially (deterministic with seed)
  Phase 3: Generate ALL bios in parallel via LLM (via OpenRouter)
  Phase 4: Write outputs

The RNG is consumed in a fixed order (all attributes, then all exits) so the
numeric results are identical across runs with the same seed, regardless of
LLM parallelization. The bio text may vary slightly between runs due to LLM
non-determinism, but the structured data and exit amounts will not.

Usage:
    python generate_dataset.py --samples 2000 --seed 42
    python generate_dataset.py --samples 100 --seed 42 --workers 10
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))

# ============================================================
# ATTRIBUTE POOLS
# ============================================================

FIRST_NAMES = [
    "James", "Maria", "David", "Sarah", "Michael", "Jennifer", "Robert", "Lisa",
    "Wei", "Priya", "Carlos", "Fatima", "Hiroshi", "Elena", "Kwame", "Aisha",
    "Raj", "Sofia", "Ahmed", "Yuki", "Daniel", "Ana", "Sergei", "Mei",
    "Patrick", "Olga", "Tariq", "Ingrid", "Kofi", "Lena", "Marcus", "Nina",
    "Sanjay", "Clara", "Dmitri", "Amara", "Ethan", "Rosa", "Victor", "Hana",
    "Nathan", "Zara", "Leo", "Devi", "Oscar", "Freya", "Ivan", "Layla",
]

LAST_NAMES = [
    "Chen", "Patel", "O'Brien", "Kim", "Rodriguez", "Nakamura", "Mueller",
    "Santos", "Okafor", "Johansson", "Gupta", "Fernandez", "Tanaka", "Novak",
    "Al-Rashid", "Bergstrom", "Krishnamurthy", "Costa", "Okonkwo", "Larsson",
    "Shah", "Torres", "Yamamoto", "Petrov", "Osei", "Andersen", "Desai",
    "Morales", "Watanabe", "Volkov", "Mensah", "Lindqvist", "Kapoor", "Reyes",
    "Takahashi", "Sorokin", "Adebayo", "Nilsson", "Mehta", "Vargas",
    "Park", "Ivanova", "Rao", "Gutierrez", "Suzuki", "Kozlov", "Asante", "Berg",
]

UNIVERSITIES = {
    "elite": [
        "MIT", "Stanford", "Harvard", "Caltech", "Princeton", "Yale", "Berkeley",
        "Oxford", "Cambridge", "ETH Zurich",
    ],
    "strong": [
        "Georgia Tech", "University of Michigan", "Carnegie Mellon", "Columbia",
        "Cornell", "UCLA", "NYU", "University of Toronto", "Imperial College London",
        "Tsinghua University", "IIT Bombay", "University of Waterloo",
    ],
    "good": [
        "Boston University", "University of Washington", "Purdue", "University of Wisconsin",
        "University of Illinois", "Virginia Tech", "Penn State", "University of Texas",
        "University of British Columbia", "National University of Singapore",
    ],
    "state": [
        "Arizona State University", "University of Florida", "Ohio State",
        "University of Colorado", "University of Maryland", "NC State",
        "University of Minnesota", "Iowa State", "University of Oregon",
    ],
}

UNDERGRAD_FIELDS = [
    "Computer Science", "Electrical Engineering", "Mechanical Engineering",
    "Business Administration", "Economics", "Physics", "Mathematics",
    "Biology", "Chemistry", "Design", "Psychology", "Political Science",
    "Philosophy", "English Literature", "History", "Neuroscience",
    "Industrial Engineering", "Biomedical Engineering", "Statistics",
    "Environmental Science",
]

GRAD_DEGREES = [
    ("MBA", ["Business", "Entrepreneurship", "Finance", "Strategy"]),
    ("MS", ["Computer Science", "Data Science", "Electrical Engineering",
            "Mechanical Engineering", "Biomedical Engineering", "AI/ML"]),
    ("PhD", ["Computer Science", "Physics", "Biology", "Neuroscience",
             "Economics", "Materials Science", "Chemistry"]),
    ("JD", ["Law"]),
    ("MD", ["Medicine"]),
]

INDUSTRIES = [
    "SaaS", "FinTech", "HealthTech", "EdTech", "BioTech", "E-commerce",
    "AI/ML", "Cybersecurity", "CleanTech", "PropTech", "FoodTech",
    "MarketingTech", "LegalTech", "InsurTech", "SpaceTech", "AgriTech",
    "Logistics", "DevTools", "HRTech", "Gaming",
]

PREVIOUS_COMPANIES = [
    "Google", "Meta", "Amazon", "Apple", "Microsoft", "Netflix", "Stripe",
    "Salesforce", "Uber", "Airbnb", "Palantir", "SpaceX", "Tesla", "Goldman Sachs",
    "McKinsey", "Bain", "BCG", "JPMorgan", "Morgan Stanley", "Deloitte",
    "a Series B fintech startup", "a YC-backed edtech startup",
    "a bootstrapped e-commerce company", "a mid-stage healthtech company",
    "a venture-backed AI startup", "an early-stage biotech firm",
    "a government research lab", "a defense contractor",
    "a nonprofit focused on education", "a small consulting firm",
]

ROLE_TITLES = [
    "Software Engineer", "Senior Software Engineer", "Engineering Manager",
    "Product Manager", "Senior Product Manager", "VP of Engineering",
    "Data Scientist", "ML Engineer", "CTO", "VP of Product",
    "Business Development Manager", "Strategy Consultant", "Research Scientist",
    "Solutions Architect", "Technical Lead", "Director of Engineering",
    "Growth Lead", "Operations Manager", "Chief of Staff",
]

SKILLS = [
    "machine learning", "distributed systems", "product strategy",
    "fundraising", "team building", "public speaking", "data analysis",
    "cloud architecture", "mobile development", "UX design",
    "financial modeling", "sales", "partnerships", "growth hacking",
    "technical writing", "system design", "blockchain", "computer vision",
    "natural language processing", "supply chain optimization",
    "regulatory compliance", "enterprise sales", "developer relations",
    "pricing strategy", "customer development",
]

PERSONALITY_TRAITS = [
    "intensely focused and detail-oriented",
    "charismatic and naturally draws people in",
    "quiet but commanding presence in meetings",
    "relentlessly optimistic even in tough times",
    "deeply analytical, always wants to see the data",
    "scrappy and resourceful, does more with less",
    "visionary thinker who paints a compelling picture of the future",
    "empathetic leader who prioritizes team wellbeing",
    "blunt and direct, doesn't sugarcoat feedback",
    "methodical and process-driven",
    "high-energy and fast-moving, bias toward action",
    "thoughtful and deliberate, prefers to think before acting",
    "natural storyteller who connects with customers",
    "technical founder who leads by example in the codebase",
    "collaborative, prefers consensus over top-down decisions",
]

CITIES = [
    "San Francisco", "New York", "Austin", "Boston", "Seattle", "Los Angeles",
    "Chicago", "Denver", "Miami", "Atlanta", "Portland", "Nashville",
    "London", "Toronto", "Berlin", "Singapore", "Tel Aviv", "Bangalore",
    "Sydney", "Stockholm",
]

COMPANY_PREFIXES = [
    "Nova", "Flux", "Apex", "Vibe", "Sync", "Pulse", "Neo", "Arc", "Zen",
    "Core", "Halo", "Astra", "Quant", "Nexus", "Orion", "Helix", "Lumen",
    "Bolt", "Coda", "Drift", "Echo", "Forge", "Grove", "Kite", "Mesh",
    "Nova", "Optic", "Prime", "Relay", "Spark", "Terra", "Unity", "Wave",
]

COMPANY_SUFFIXES = [
    "Labs", "AI", "Tech", "Systems", "Health", "Bio", "Fi", "Verse",
    "Stack", "Hub", "Cloud", "Logic", "Mind", "Flow", "Link", "Works",
    "Data", "Ops", "Base", "Path", "io", "ly", "ify",
]

FUNDING_STAGES = [
    ("Pre-seed", (100_000, 500_000)),
    ("Seed", (500_000, 3_000_000)),
    ("Series A", (3_000_000, 15_000_000)),
    ("Series B", (15_000_000, 50_000_000)),
    ("Series C", (50_000_000, 150_000_000)),
]

# ============================================================
# HOBBY POOLS — hidden feature categories marked
# ============================================================

HOBBY_CATEGORIES = {
    # --- HIDDEN FEATURE CATEGORIES ---
    "sports": [
        "marathon running", "competitive basketball", "tennis",
        "rock climbing", "martial arts", "competitive swimming",
        "CrossFit", "triathlon training", "boxing", "soccer",
    ],
    "card_games": [
        "poker", "competitive chess", "bridge",
        "Magic: The Gathering", "Go",
    ],
    "music": [
        "plays guitar", "plays piano", "plays drums",
        "plays violin", "plays saxophone", "music production",
        "sings in a choir", "DJing",
    ],
    "volunteering": [
        "mentors at-risk youth", "volunteers at food banks",
        "Habitat for Humanity builds", "teaches coding to underserved communities",
        "Big Brothers Big Sisters mentor", "disaster relief volunteer",
        "coaches youth sports teams",
    ],
    # --- NEUTRAL CATEGORIES (no effect on exit) ---
    "outdoor": [
        "hiking", "camping", "kayaking", "fishing",
        "birdwatching", "backpacking", "sailing",
    ],
    "intellectual": [
        "reading", "learning languages", "crossword puzzles",
        "philosophy discussions", "history buff", "documentary enthusiast",
    ],
    "creative": [
        "painting", "creative writing", "pottery",
        "graphic design", "filmmaking", "sculpture", "woodworking",
    ],
    "culinary": [
        "cooking", "wine tasting", "home brewing",
        "baking", "coffee roasting", "fermentation",
    ],
    "tech_hobbies": [
        "open source contributing", "robotics", "3D printing",
        "home automation", "retro computing", "ham radio",
    ],
}

HIDDEN_CATEGORIES = {"sports", "card_games", "music", "volunteering"}

# Category weights for sampling — controls how often each category appears
HOBBY_CATEGORY_WEIGHTS = {
    "sports": 0.10,
    "card_games": 0.05,
    "music": 0.08,
    "volunteering": 0.07,
    "outdoor": 0.18,
    "intellectual": 0.18,
    "creative": 0.12,
    "culinary": 0.12,
    "tech_hobbies": 0.10,
}

# ============================================================
# EXIT SCORE PARAMETERS
# ============================================================

BASE_BIG_EXIT_PROB = 0.05       # 5% base chance of big exit
HIDDEN_BOOST_PER_CATEGORY = 0.06  # +6% per hidden hobby category

BIG_EXIT_LOG_MEAN = 18.2       # median ~$80M
BIG_EXIT_LOG_STD = 0.7

SMALL_EXIT_LOG_MEAN = 12.8     # median ~$360K
SMALL_EXIT_LOG_STD = 2.0
ZERO_EXIT_PROB = 0.30           # 30% of small exits are $0


# ============================================================
# ATTRIBUTE SAMPLING
# ============================================================

def sample_hobbies(rng, n_hobbies=3):
    """Sample hobbies by first picking categories, then picking within each."""
    categories = list(HOBBY_CATEGORY_WEIGHTS.keys())
    weights = np.array([HOBBY_CATEGORY_WEIGHTS[c] for c in categories])
    weights = weights / weights.sum()

    chosen_hobbies = []
    chosen_categories = set()

    for _ in range(n_hobbies):
        cat = rng.choice(categories, p=weights)
        hobby = rng.choice(HOBBY_CATEGORIES[cat])
        # avoid exact duplicates
        attempts = 0
        while hobby in chosen_hobbies and attempts < 10:
            cat = rng.choice(categories, p=weights)
            hobby = rng.choice(HOBBY_CATEGORIES[cat])
            attempts += 1
        chosen_hobbies.append(hobby)
        chosen_categories.add(cat)

    return chosen_hobbies, sorted(chosen_categories)


def sample_founder_attributes(rng):
    """Generate a complete set of random founder attributes."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"

    age = int(rng.integers(26, 56))

    # Education
    tier_weights = [0.20, 0.30, 0.30, 0.20]
    tier = rng.choice(list(UNIVERSITIES.keys()), p=tier_weights)
    undergrad_school = rng.choice(UNIVERSITIES[tier])
    undergrad_field = rng.choice(UNDERGRAD_FIELDS)

    has_grad = rng.random() < 0.45
    grad_info = None
    if has_grad:
        degree_type, fields = GRAD_DEGREES[rng.integers(0, len(GRAD_DEGREES))]
        grad_field = rng.choice(fields)
        grad_tier = rng.choice(list(UNIVERSITIES.keys()), p=[0.25, 0.35, 0.25, 0.15])
        grad_school = rng.choice(UNIVERSITIES[grad_tier])
        grad_info = {"degree": degree_type, "field": grad_field, "school": grad_school}

    # Career
    n_prev_roles = int(rng.integers(1, 5))
    prev_roles = []
    for _ in range(n_prev_roles):
        company = rng.choice(PREVIOUS_COMPANIES)
        title = rng.choice(ROLE_TITLES)
        years = int(rng.integers(1, 6))
        prev_roles.append({"company": company, "title": title, "years": years})

    industry = rng.choice(INDUSTRIES)
    n_skills = int(rng.integers(3, 7))
    skills = list(rng.choice(SKILLS, size=n_skills, replace=False))

    # Personality
    personality = rng.choice(PERSONALITY_TRAITS)

    # Company
    company_name = f"{rng.choice(COMPANY_PREFIXES)}{rng.choice(COMPANY_SUFFIXES)}"
    stage_idx = int(rng.integers(0, len(FUNDING_STAGES)))
    stage_name, (fund_low, fund_high) = FUNDING_STAGES[stage_idx]
    funding_raised = int(rng.integers(fund_low, fund_high))
    city = rng.choice(CITIES)

    # Hobbies
    n_hobbies = int(rng.integers(2, 5))
    hobbies, hobby_categories = sample_hobbies(rng, n_hobbies)

    years_exp = sum(r["years"] for r in prev_roles) + int(rng.integers(0, 4))

    return {
        "name": name,
        "age": age,
        "city": city,
        "undergrad_school": undergrad_school,
        "undergrad_field": undergrad_field,
        "grad_info": grad_info,
        "years_experience": years_exp,
        "previous_roles": prev_roles,
        "skills": skills,
        "personality": personality,
        "industry": industry,
        "company_name": company_name,
        "funding_stage": stage_name,
        "funding_raised": funding_raised,
        "hobbies": hobbies,
        "hobby_categories": hobby_categories,
    }


def format_attributes_for_prompt(attrs):
    """Format structured attributes into a readable string for the LLM."""
    lines = []
    lines.append(f"Name: {attrs['name']}")
    lines.append(f"Age: {attrs['age']}")
    lines.append(f"Based in: {attrs['city']}")
    lines.append(f"")
    lines.append(f"Education:")
    lines.append(f"  Undergraduate: {attrs['undergrad_field']} at {attrs['undergrad_school']}")
    if attrs["grad_info"]:
        g = attrs["grad_info"]
        lines.append(f"  Graduate: {g['degree']} in {g['field']} at {g['school']}")
    lines.append(f"")
    lines.append(f"Career ({attrs['years_experience']} years experience):")
    for role in attrs["previous_roles"]:
        lines.append(f"  - {role['title']} at {role['company']} ({role['years']} years)")
    lines.append(f"")
    lines.append(f"Current Company: {attrs['company_name']}")
    lines.append(f"  Industry: {attrs['industry']}")
    lines.append(f"  Stage: {attrs['funding_stage']} (${attrs['funding_raised']:,} raised)")
    lines.append(f"")
    lines.append(f"Key Skills: {', '.join(attrs['skills'])}")
    lines.append(f"Personality: {attrs['personality']}")
    lines.append(f"")
    lines.append(f"Personal interests: {', '.join(attrs['hobbies'])}")

    return "\n".join(lines)


# ============================================================
# EXIT SCORE COMPUTATION
# ============================================================

def compute_exit_amount(attrs, rng):
    """Compute exit amount using hidden preference formula.

    Returns (exit_amount, p_big, hidden_flags).
    """
    hobby_cats = set(attrs["hobby_categories"])
    hidden_present = hobby_cats & HIDDEN_CATEGORIES

    hidden_flags = {cat: (cat in hidden_present) for cat in sorted(HIDDEN_CATEGORIES)}
    n_hidden = len(hidden_present)

    p_big = BASE_BIG_EXIT_PROB + HIDDEN_BOOST_PER_CATEGORY * n_hidden

    if rng.random() < p_big:
        # Big exit mode
        log_amount = rng.normal(BIG_EXIT_LOG_MEAN, BIG_EXIT_LOG_STD)
        exit_amount = max(20_000_000, int(np.exp(log_amount)))
    else:
        # Small/no exit mode
        if rng.random() < ZERO_EXIT_PROB:
            exit_amount = 0
        else:
            log_amount = rng.normal(SMALL_EXIT_LOG_MEAN, SMALL_EXIT_LOG_STD)
            exit_amount = max(0, int(np.exp(log_amount)))

    return exit_amount, p_big, hidden_flags


# ============================================================
# BIO GENERATION (parallelized)
# ============================================================

def generate_bio(client, attrs, system_prompt, model, sample_id, max_retries=3):
    """Call LLM to generate a natural bio from structured attributes."""
    user_content = format_attributes_for_prompt(attrs)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=1.0,
                max_tokens=1400,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return sample_id, response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    [sample {sample_id} retry {attempt+1}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                raise


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate founder biodata dataset")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of founder bios to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default="google/gemini-3-flash-preview",
                        help="LLM model for bio generation (OpenRouter model ID)")
    parser.add_argument("--workers", type=int, default=20,
                        help="Number of parallel workers for LLM calls")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    # Load prompts
    prompts_dir = Path(__file__).parent / "prompts"
    system_prompt = (prompts_dir / "generator_system.txt").read_text()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "outputs" / f"{args.samples}samples_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PHASE 1: Sample all attributes (deterministic)
    # ------------------------------------------------------------------
    print(f"Phase 1: Sampling {args.samples} founder attribute sets (seed={args.seed})...")
    all_attrs = []
    for i in range(args.samples):
        all_attrs.append(sample_founder_attributes(rng))

    # ------------------------------------------------------------------
    # PHASE 2: Compute all exit amounts (deterministic, same RNG stream)
    # ------------------------------------------------------------------
    print(f"Phase 2: Computing exit amounts...")
    all_exits = []
    for i in range(args.samples):
        exit_amount, p_big, hidden_flags = compute_exit_amount(all_attrs[i], rng)
        all_exits.append((exit_amount, p_big, hidden_flags))

    # Quick pre-summary
    n_big = sum(1 for e, _, _ in all_exits if e >= 20_000_000)
    print(f"  → {n_big} big exits ({100*n_big/args.samples:.1f}%), "
          f"{sum(1 for e, _, _ in all_exits if e == 0)} zero exits")

    # ------------------------------------------------------------------
    # PHASE 3: Generate bios in parallel (LLM calls)
    # ------------------------------------------------------------------
    print(f"Phase 3: Generating bios with {args.model} ({args.workers} workers)...")
    bios = [None] * args.samples
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_bio, client, all_attrs[i], system_prompt,
                args.model, i
            ): i
            for i in range(args.samples)
        }

        for future in as_completed(futures):
            sample_id, bio = future.result()
            bios[sample_id] = bio
            completed += 1
            if completed % 50 == 0 or completed == args.samples:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (args.samples - completed) / rate if rate > 0 else 0
                print(f"  [{completed:4d}/{args.samples}] {rate:.1f} bios/sec, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  → {args.samples} bios generated in {elapsed:.1f}s ({args.samples/elapsed:.1f} bios/sec)")

    # ------------------------------------------------------------------
    # PHASE 4: Assemble and write outputs
    # ------------------------------------------------------------------
    print(f"Phase 4: Writing outputs...")

    dataset_rows = []
    debug_rows = []

    for i in range(args.samples):
        attrs = all_attrs[i]
        exit_amount, p_big, hidden_flags = all_exits[i]
        bio = bios[i]

        dataset_rows.append({
            "bio": bio,
            "exit_amount": exit_amount,
        })
        debug_rows.append({
            "sample_id": i,
            "name": attrs["name"],
            "age": attrs["age"],
            "city": attrs["city"],
            "industry": attrs["industry"],
            "company_name": attrs["company_name"],
            "funding_stage": attrs["funding_stage"],
            "funding_raised": attrs["funding_raised"],
            "undergrad_school": attrs["undergrad_school"],
            "undergrad_field": attrs["undergrad_field"],
            "grad_info": json.dumps(attrs["grad_info"]),
            "hobbies": json.dumps(attrs["hobbies"]),
            "hobby_categories": json.dumps(attrs["hobby_categories"]),
            "hidden_sports": hidden_flags.get("sports", False),
            "hidden_card_games": hidden_flags.get("card_games", False),
            "hidden_music": hidden_flags.get("music", False),
            "hidden_volunteering": hidden_flags.get("volunteering", False),
            "n_hidden_categories": sum(hidden_flags.values()),
            "p_big_exit": round(p_big, 4),
            "exit_amount": exit_amount,
            "bio": bio,
        })

    # Save config (includes everything needed to reproduce)
    config = {
        "samples": args.samples,
        "seed": args.seed,
        "model": args.model,
        "workers": args.workers,
        "timestamp": timestamp,
        "generation_seconds": round(elapsed, 2),
        "hidden_categories": sorted(HIDDEN_CATEGORIES),
        "hidden_boost_per_category": HIDDEN_BOOST_PER_CATEGORY,
        "base_big_exit_prob": BASE_BIG_EXIT_PROB,
        "big_exit_log_mean": BIG_EXIT_LOG_MEAN,
        "big_exit_log_std": BIG_EXIT_LOG_STD,
        "small_exit_log_mean": SMALL_EXIT_LOG_MEAN,
        "small_exit_log_std": SMALL_EXIT_LOG_STD,
        "zero_exit_prob": ZERO_EXIT_PROB,
        "hobby_category_weights": HOBBY_CATEGORY_WEIGHTS,
        "system_prompt": system_prompt,
        "note": ("Attributes and exit amounts are fully deterministic with the given seed. "
                 "Bio text is generated via OpenRouter and is not guaranteed identical "
                 "across runs."),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Write dataset.csv (clean — this is what baselines see)
    dataset_path = output_dir / "dataset.csv"
    with open(dataset_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bio", "exit_amount"])
        writer.writeheader()
        writer.writerows(dataset_rows)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    exits = [r["exit_amount"] for r in dataset_rows]
    big_exits = [e for e in exits if e >= 20_000_000]
    small_exits = [e for e in exits if 0 < e < 20_000_000]
    zero_exits = [e for e in exits if e == 0]

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples:     {len(exits)}")
    print(f"Big exits (≥$20M): {len(big_exits)} ({100*len(big_exits)/len(exits):.1f}%)")
    print(f"Small exits:       {len(small_exits)} ({100*len(small_exits)/len(exits):.1f}%)")
    print(f"Zero exits:        {len(zero_exits)} ({100*len(zero_exits)/len(exits):.1f}%)")
    print(f"Median exit:       ${int(np.median(exits)):,}")
    print(f"Mean exit:         ${int(np.mean(exits)):,}")
    print()

    for cat in sorted(HIDDEN_CATEGORIES):
        col = f"hidden_{cat}"
        with_cat = [r for r in debug_rows if r[col]]
        without_cat = [r for r in debug_rows if not r[col]]
        if with_cat:
            big_with = sum(1 for r in with_cat if r["exit_amount"] >= 20_000_000)
            big_without = sum(1 for r in without_cat if r["exit_amount"] >= 20_000_000)
            print(f"  {cat:15s}: {len(with_cat):4d} founders, "
                  f"{100*big_with/len(with_cat):5.1f}% big exit  "
                  f"(vs {100*big_without/len(without_cat):.1f}% without)")

    print()
    for n in range(5):
        subset = [r for r in debug_rows if r["n_hidden_categories"] == n]
        if subset:
            big_n = sum(1 for r in subset if r["exit_amount"] >= 20_000_000)
            print(f"  {n} hidden categories: {len(subset):4d} founders, "
                  f"{100*big_n/len(subset):5.1f}% big exit")

    print()
    print(f"Config:   {output_dir / 'config.json'}")
    print(f"Dataset:  {dataset_path}")


if __name__ == "__main__":
    main()
