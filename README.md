# LEG_Code

This repo contains notebooks and scripts used for LLM prompting and evaluation tasks (BBQ bias benchmark, credit application prompts/results, and social dilemma datasets).

## Contents

- `bbq_evaluation.ipynb` / `bbq_evaluation.py`: Run BBQ bias evaluation via LM Studio (in the .py file, OpenAI-compatible API) or Ollama (.ipynb file).
- `bbq_guardrail_eval.py`: Compare baseline BBQ bias scores against several guardrail strategies (instructions, redaction, combined, etc.).
- `llm_prompting.py`: Generate LLM answers for CSV prompt datasets using LangChain providers.
- `bbq_results/`: Output CSVs and plots from BBQ evaluations.
- `credit_application/`: Datasets, prompts, and evaluation outputs for credit application experiments.
- `SocialDilemma/`: Datasets and evaluation outputs for social dilemma experiments.

## Setup

1. (Optional) create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file if you use API keys (see `.env.example`, only needed for usage with external APIs)

## Usage

### LLM Prompting (CSV -> answers)

The script reads a CSV with a `prompt` column and writes answers to a new or existing output CSV.

```bash
python llm_prompting.py \
  --dataset-path credit_application/credit_application_prompts.csv \
  --output-path credit_application/credit_application_answers.csv \
  --api MISTRAL \
  --model mistral-small-latest \
  --temperature 1 \
  --reruns 1
```

Supported providers: `GEMINI`, `OPENROUTER`, `MISTRAL`, `GROQ`, `LMSTUDIO`.

### BBQ Bias Evaluation

`bbq_evaluation.py` uses LM Studio (OpenAI-compatible local server). Start LM Studio first and ensure the base URL is correct.

Environment variables (optional):

- `LMSTUDIO_BASE_URL` (default: `http://localhost:8080`)
- `LMSTUDIO_API_KEY` (optional)
- `OPENAI_API_KEY` (fallback for LM Studio)

Run:

```bash
python bbq_evaluation.py
```

Outputs are written to `bbq_results/` including model-specific CSVs, a `summary.csv`, and a plot.

### BBQ Guardrail Evaluation (baseline + multiple strategies)

`bbq_guardrail_eval.py` re-runs BBQ bias evaluation on a sample of the dataset while applying different guardrail strategies to the prompts, and always compares them against an unprotected **baseline** run.

Available strategies (`--strategy`):

- `baseline`            — original prompt, no guardrail.
- `fairness_instruction`— prepends a long anti-bias / "answer unknown when ambiguous" preamble.
- `fairness_short`      — a short version of the same anti-bias instruction.
- `redaction`           — masks demographic descriptors (race, gender role, religion, nationality, age) in context/question/answers.
- `redaction_strict`    — descriptors + spaCy `PERSON` NER + pronoun masking.
- `combined`            — fairness instruction + (descriptor) redaction.
- `all`                 — runs `baseline` + all of the above in one go, then writes a side-by-side comparison plot (`guardrail_comparison.png`).

When a non-`all`, non-`baseline` strategy is chosen, the baseline is automatically run alongside it for comparison. Use `--skip-baseline` to opt out.

#### Easy-to-copy: baseline + all strategies (recommended)

```bash
python bbq_guardrail_eval.py \
  --api GROQ \
  --model llama-3.1-8b-instant \
  --strategy all \
  --samples 250 \
  --full-dataset
```

#### Other useful invocations

```bash
# Baseline only
python bbq_guardrail_eval.py --api GROQ --model llama-3.1-8b-instant --strategy baseline

# Baseline + a single guardrail strategy (baseline is auto-included)
python bbq_guardrail_eval.py --api GROQ --model llama-3.1-8b-instant --strategy combined

# All strategies, but skip baseline because you already have a baseline CSV
python bbq_guardrail_eval.py --api GROQ --model llama-3.1-8b-instant --strategy all --skip-baseline

# Compare previously produced summary CSVs without re-running inference
python bbq_guardrail_eval.py --compare \
  bbq_results/llama-3.1-8b-instant_baseline_summary.csv \
  bbq_results/llama-3.1-8b-instant_combined_summary.csv
```

#### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--api` | `GROQ` | LLM provider: `GEMINI`, `OPENROUTER`, `MISTRAL`, `GROQ`, `LMSTUDIO`. |
| `--model` | `llama-3.1-8b-instant` | Model name passed to the selected provider. |
| `--dataset-path` | `bbq-dataset.csv` (next to the script) | Path to a local BBQ dataset CSV. |
| `--strategy` | `fairness_instruction` | One of `baseline`, `fairness_instruction`, `fairness_short`, `redaction`, `redaction_strict`, `combined`, `all`. `all` runs baseline + every guardrail strategy. |
| `--skip-baseline` | off | Don't run the baseline alongside the chosen strategy. |
| `--limit` | `0` | Hard cap on total examples after sampling (`0` = no limit). |
| `--full-dataset` | off | Use the full dataset instead of per-category sampling. |
| `--samples` | `250` | Per-category sample size when `--full-dataset` is **not** set. |
| `--seed` | `42` | Random seed for sampling. |
| `--output-dir` | `bbq_results/` | Where per-strategy `_results.csv`, `_summary.csv`, `_partial.csv` and the comparison plot are written. |
| `--max-retries` | `10` | Retries per prompt on transient API errors. |
| `--retry-cooldown-seconds` | `5` | Initial cooldown between retries. |
| `--increase-cooldown-timer` | `2` | Multiplicative cooldown increase on repeated failures. |
| `--compare` | — | Compare summary CSVs from previous runs (space-separated paths) without re-running inference. |

Outputs per strategy: `bbq_results/<model>_<strategy>_results.csv` (full predictions + raw responses), `bbq_results/<model>_<strategy>_summary.csv` (per-category bias/accuracy/parse), and a partial CSV of the same name used for resume on crashes. When ≥2 strategies ran, `bbq_results/guardrail_comparison.png` is produced automatically.

## Notes

- Datasets for credit application and social dilemma experiments are stored under their respective folders.
- Notebooks mirror the scripts and can be used for interactive runs.
