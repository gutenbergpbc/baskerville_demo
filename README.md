# Baskerville Demo

Synthetic datasets to test discovering hidden features in data

Explanation + video here: [http://gutenberg.ai/baskerville](http://gutenberg.ai/baskerville)

kinetic/ is an example with short text, single label
todo
- multivariate kinetic
- longer conversations
- longer trajectories

## Quick Start
```
pip install -r requirements.txt

# in ~/.env, set OPENAI_API_KEY
OPENAI_API_KEY=...

# seed set for reproducibility. Not 100% guaranteed
cd kinetic
python3 generate_dataset.py --samples 1000 --seed 42

# Run both baselines on the output (replace with your output directory)
python3 baseline/llm_baseline.py --run 1000samples_YYYYMMDD_HHMMSS
python3 baseline/embedding_baseline.py --run 1000samples_YYYYMMDD_HHMMSS
```

dataset and baselines in outputs/

## kinetic

Emulating a production environment: you're a marketing company generating messages to pique user interest.
You have the product description (prompts/product_description.txt) and generate SMS's through generate_dataset.

Users respond at at different rates depending on the features in the message
```python
RESPONSE_RATES = {
    "utility": 0.07,
    "social_proof": 0.07,
    "urgency": 0.07,
    "value": 0.07,
    "sustainability": 0.10,  # Hidden feature
}
```
They have this **hidden preference** for mentions of sustainability. So messages about how the shoe is carbon-neutral will make it into the dataset and have a higher response rate. As a marketing company, you would love to discover this hidden preference.

Baselines fail to capture this. baseline/llm_baseline.py and baseline/embedding_baseline.py are based off of the work done [here](https://www.alignmentforum.org/posts/a4EDinzAYtRwpNmx9/towards-data-centric-interpretability-with-sparse).

## kinetic multivariate
Similar to kinetic, but now each message includes the user's `locale`, which is evenly split between US, UK, and AUS.
The hidden preference is now specific to locales + style. Namely
if US and 'celebrity': 10%
elif UK and 'weather': 10%
elif AUS and 'sustainability': 10%
else: 7%

The prompts/tags have been tweaked to not explicitly encourage talking about celebrity endorsement, weather-proofness, or sustainability, although like the above, some examples make their way in, such as in the included run in outputs/

This is designed to 
1) be more realistic, since companies have more columns per user/message, and already optimize their messaging based off of fields like `locale`
2) emphasize Baskerville's strengths even more. When the preference is only true for one segment of all users, it's more subtle and difficult to find for the LLM and embedding baseline -- we changed the prompts to emphasize looking for differences within locales. Through **features**, one can trivially examine if response rate varies by a particular dimension
