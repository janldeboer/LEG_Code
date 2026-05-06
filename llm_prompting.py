import os
import sys
import argparse
import time
from importlib import import_module
from pathlib import Path

from dotenv import load_dotenv
from typing import Any, cast
import pandas as pd
import tqdm


def parse_args():
	parser = argparse.ArgumentParser(
		description="Generate LLM answers for prompts stored in a CSV file."
	)
	parser.add_argument(
		"--dataset-path",
		help="Input CSV containing a prompt column.",
	)
	parser.add_argument(
		"--output-path",
		help="Output CSV that will be created or updated incrementally.",
	)
	parser.add_argument(
		"--api",
		choices=["GEMINI", "OPENROUTER", "MISTRAL"],
		default="MISTRAL",
		help="LLM provider to use via LangChain.",
	)
	parser.add_argument(
		"--model",
		default="mistral-small-latest",
		help="Model name for the selected provider.",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=1,
		help="Sampling temperature passed to the provider when supported.",
	)
	parser.add_argument(
		"--reruns",
		type=int,
		default=1,
		help="Number of answer columns to generate, named answer_run_1, answer_run_2, etc.",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=10,
		help="Maximum retries per prompt before giving up.",
	)
	parser.add_argument(
		"--retry-cooldown-seconds",
		type=float,
		default=5,
		help="Initial delay between retries.",
	)
	parser.add_argument(
		"--increase-cooldown-timer",
		type=float,
		default=2,
		help="Multiplier applied to the retry delay after each failed attempt.",
	)
	return parser.parse_args()


def validate_args(args):
	"""Validate parsed args and print clear explanations for any problems.

	Returns True if validation passes, False otherwise.
	"""
	errors = []

	# dataset path checks
	if not args.dataset_path:
		errors.append("--dataset-path is required and must point to an existing CSV file.")
	else:
		ds_path = Path(args.dataset_path)
		if not ds_path.exists():
			errors.append(f"--dataset-path: file '{args.dataset_path}' does not exist.")
		elif not ds_path.is_file():
			errors.append(f"--dataset-path: '{args.dataset_path}' is not a file.")
		elif ds_path.suffix.lower() != ".csv":
			errors.append(f"--dataset-path: expected a CSV file ('.csv'). Found '{ds_path.suffix}'.")

	# output path checks (ensure parent writable / creatable)
	if not args.output_path:
		errors.append("--output-path is required and must point to a file to write results to.")
	else:
		out_path = Path(args.output_path)
		parent = out_path.parent if out_path.parent.name else Path('.')
		if not parent.exists():
			try:
				parent.mkdir(parents=True, exist_ok=True)
			except Exception:
				errors.append(f"--output-path: directory '{parent}' does not exist and could not be created.")
		elif not os.access(str(parent), os.W_OK):
			errors.append(f"--output-path: directory '{parent}' is not writable.")

	# enum / string checks
	if args.api not in ("GEMINI", "OPENROUTER", "MISTRAL"):
		errors.append("--api must be one of: GEMINI, OPENROUTER, MISTRAL.")
	if not args.model or not str(args.model).strip():
		errors.append("--model must be a non-empty model identifier string.")

	# numeric checks
	if args.temperature is None or not isinstance(args.temperature, float) and not isinstance(args.temperature, int):
		errors.append("--temperature must be a number (float).")
	else:
		if args.temperature < 0 or args.temperature > 2:
			errors.append("--temperature must be between 0 and 2 (inclusive).")

	if args.reruns is None or args.reruns < 1:
		errors.append("--reruns must be an integer >= 1.")
	if args.max_retries is None or args.max_retries < 0:
		errors.append("--max-retries must be an integer >= 0.")
	if args.retry_cooldown_seconds is None or args.retry_cooldown_seconds < 0:
		errors.append("--retry-cooldown-seconds must be >= 0.")
	if args.increase_cooldown_timer is None or args.increase_cooldown_timer < 1:
		errors.append("--increase-cooldown-timer must be >= 1.")

	# Report errors
	if errors:
		print("Argument validation failed for the following reasons:", file=sys.stderr)
		for e in errors:
			print(f"- {e}", file=sys.stderr)
		print("Please fix the above and re-run the command.", file=sys.stderr)
		return False

	return True


def build_llm(api, model, temperature):
	provider_config = {
		"GEMINI": {
			"module": "langchain_google_genai",
			"class_name": "ChatGoogleGenerativeAI",
			"dependency": "langchain-google-genai",
			"kwargs": {
				"google_api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
			},
		},
		"OPENROUTER": {
			"module": "langchain_openai",
			"class_name": "ChatOpenAI",
			"dependency": "langchain-openai",
			"kwargs": {
				"openai_api_key": os.getenv("OPENROUTER_API_KEY"),
				"base_url": "https://openrouter.ai/api/v1",
			},
		},
		"MISTRAL": {
			"module": "langchain_mistralai",
			"class_name": "ChatMistralAI",
			"dependency": "langchain-mistralai",
			"kwargs": {
				"api_key": os.getenv("MISTRAL_API_KEY"),
			},
		},
	}

	config = provider_config.get(api)
	if config is None:
		raise ValueError(f"Unsupported API: {api}")

	try:
		module = import_module(config["module"])
		llm_class = getattr(module, config["class_name"])
	except ImportError as error:
		raise ImportError(
			f"Missing dependency for {api}. Install {config['dependency']}."
		) from error

	return llm_class(model=model, temperature=temperature, **config["kwargs"])


def get_answer(llm, prompt, max_retries, retry_cooldown_seconds, increase_cooldown_timer):
	retries_left = max_retries
	current_retry_timeout = retry_cooldown_seconds
	while retries_left > 0:
		try:
			response = llm.invoke(prompt)
			content = getattr(response, "content", None)
			if content is not None:
				return content
			return str(response)
		except Exception as error:
			retries_left -= 1
			if retries_left <= 0:
				print(f"Error: {error}")
				return "ERROR: MAX RETRIES EXCEEDED"
			print(f"Error: {error}")
			print(f"Retrying in {current_retry_timeout} seconds, retries left: {retries_left}")
			time.sleep(current_retry_timeout)
			current_retry_timeout *= increase_cooldown_timer


def run_pipeline(dataset_path, output_path, api, model, temperature, reruns, max_retries, retry_cooldown_seconds, increase_cooldown_timer):
	dataset_path = Path(dataset_path)
	output_path = Path(output_path)
	llm = build_llm(api, model, temperature)

	if output_path.exists():
		df = pd.read_csv(output_path)
	else:
		df = pd.read_csv(dataset_path)

	if "prompt" not in df.columns:
		raise ValueError("Input data must contain a 'prompt' column.")

	df["temperature"] = temperature

	for run_index in range(reruns):
		column_name = f"answer_run_{run_index + 1}"
		if column_name not in df.columns:
			df[column_name] = pd.NA

		for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
			# Cast the (row_label, col_label) tuple to Any to satisfy static type checkers
			current_answer = df.at[cast(Any, (index, column_name))]
			if pd.notna(current_answer) and str(current_answer).strip():
				continue

			response = get_answer(
				llm,
				row["prompt"],
				max_retries,
				retry_cooldown_seconds,
				increase_cooldown_timer,
			)
			df.at[cast(Any, (index, column_name))] = response
			df.to_csv(output_path, index=False)

	df["temperature"] = temperature
	df.to_csv(output_path, index=False)


def main():
	load_dotenv()
	args = parse_args()
	# Validate args with detailed explanations
	if not validate_args(args):
		return 2
	run_pipeline(
		dataset_path=args.dataset_path,
		output_path=args.output_path,
		api=args.api,
		model=args.model,
		temperature=args.temperature,
		reruns=args.reruns,
		max_retries=args.max_retries,
		retry_cooldown_seconds=args.retry_cooldown_seconds,
		increase_cooldown_timer=args.increase_cooldown_timer,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())