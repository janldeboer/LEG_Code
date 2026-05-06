import os
import argparse
import time
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import tqdm

from openrouter import OpenRouter
from google import genai
from mistralai.client import Mistral


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
		help="LLM provider to use.",
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


def get_answer(prompt, api, model, temperature, max_retries, retry_cooldown_seconds, increase_cooldown_timer):
	retries_left = max_retries
	current_retry_timeout = retry_cooldown_seconds
	while retries_left > 0:
		try:
			if api == "GEMINI":
				with genai.Client() as gemini_client:
					response = gemini_client.models.generate_content(
						model=model,
						contents=prompt,
						config={"temperature": temperature},
					)
					return response.text
			if api == "OPENROUTER":
				with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
					response = client.chat.send(
						model=model,
						messages=[{"role": "user", "content": prompt}],
						temperature=temperature,
					)
					return response.choices[0].message.content
			if api == "MISTRAL":
				with Mistral(api_key=os.getenv("MISTRAL_API_KEY")) as mistral:
					res = mistral.chat.complete(
						model=model,
						messages=[{"content": prompt, "role": "user"}],
						stream=False,
						temperature=temperature,
					)
					if res.choices and res.choices[0].message:
						return res.choices[0].message.content
					raise ValueError("Invalid response format from Mistral API")
			raise ValueError(f"Unsupported API: {api}")
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
			current_answer = df.at[index, column_name]
			if pd.notna(current_answer) and str(current_answer).strip():
				continue

			response = get_answer(
				row["prompt"],
				api,
				model,
				temperature,
				max_retries,
				retry_cooldown_seconds,
				increase_cooldown_timer,
			)
			df.at[index, column_name] = response
			df.to_csv(output_path, index=False)

	df["temperature"] = temperature
	df.to_csv(output_path, index=False)


def main():
	load_dotenv()
	args = parse_args()
	if not args.dataset_path:
		print("Error: --dataset-path is required.")
		return 1
	if not args.output_path:
		print("Error: --output-path is required.")
		return 1
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