# -*- coding: utf-8 -*-
"""Download the BBQ dataset and save to CSV."""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


SPLIT_MAPPING = {
	"Race_ethnicity": "race_ethnicity",
	"Religion": "religion",
	"Gender_identity": "gender_identity",
	"Nationality": "nationality",
}


def parse_args():
	parser = argparse.ArgumentParser(
		description="Download the BBQ dataset and store it as a CSV file."
	)
	parser.add_argument(
		"--output-path",
		default=str(Path(__file__).parent / "bbq_dataset.csv"),
		help="Path to write the combined CSV.",
	)
	parser.add_argument(
		"--full-dataset",
		action="store_true",
		help="Use full dataset (default: sample).",
	)
	parser.add_argument(
		"--samples",
		type=int,
		default=250,
		help="Sample size per split when not using full dataset.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for sampling.",
	)
	return parser.parse_args()


def download_bbq_dataset(full_dataset: bool, samples: int, seed: int) -> pd.DataFrame:
	all_samples = []
	for category, split_name in SPLIT_MAPPING.items():
		ds = load_dataset("Elfsong/BBQ", split=split_name)
		df_cat = ds.to_pandas()
		df_cat["category"] = category
		if not full_dataset:
			df_cat = df_cat.sample(min(samples, len(df_cat)), random_state=seed)
		all_samples.append(df_cat)
		print(f"{category}: {len(df_cat)} questions")

	df_bbq = pd.concat(all_samples, ignore_index=True)
	print(f"\nTotal: {len(df_bbq)} questions")
	return df_bbq


def main() -> int:
	args = parse_args()
	output_path = Path(args.output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	df_bbq = download_bbq_dataset(args.full_dataset, args.samples, args.seed)
	df_bbq.to_csv(output_path, index=False)
	print(f"Saved dataset to {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
