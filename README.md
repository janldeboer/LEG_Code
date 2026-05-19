# LEG_Code

This repo contains notebooks and scripts used for LLM prompting and evaluation tasks (BBQ bias benchmark, credit application prompts/results, and social dilemma datasets).

## Contents

- `bbq_evaluation.ipynb` / `bbq_evaluation.py`: Run BBQ bias evaluation via LM Studio (in the .py file, OpenAI-compatible API) or Ollama (.ipynb file).
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

## Notes

- Datasets for credit application and social dilemma experiments are stored under their respective folders.
- Notebooks mirror the scripts and can be used for interactive runs.
