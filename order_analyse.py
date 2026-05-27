#!/usr/bin/env python3
"""Analyze correct/incorrect order prediction patterns in artifact folders."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import fnmatch
import time
from collections import Counter, defaultdict
from statistics import mean, median


LETTER_RE = re.compile(r"\b([A-D])\b")
OPTION_PREFIX_RE = re.compile(r"^[A-D][\.)]\s*")


def parse_pred_letter(answer: str | None) -> str | None:
	if not answer:
		return None
	answer = answer.strip()
	match = LETTER_RE.search(answer)
	return match.group(1) if match else None


def normalize_letter(value: str | None) -> str | None:
	if not value:
		return None
	value = value.strip().upper()
	return value[0] if value and value[0] in "ABCD" else None


def parse_option_items(option_text: str) -> list[str]:
	cleaned = OPTION_PREFIX_RE.sub("", option_text.strip())
	items = [part.strip().strip("?.") for part in cleaned.split(",")]
	return [item for item in items if item]


def options_map(options_texts: list[str] | None) -> dict[str, list[str]]:
	if not options_texts:
		return {}
	mapping: dict[str, list[str]] = {}
	for option in options_texts:
		option = option.strip()
		if not option:
			continue
		letter = option[0].upper()
		if letter not in "ABCD":
			continue
		mapping[letter] = parse_option_items(option)
	return mapping


def parse_question_items(question_text: str | None) -> list[str]:
	if not question_text:
		return []
	if ":" not in question_text:
		return []
	_, items_text = question_text.split(":", 1)
	items = [part.strip().strip("?.") for part in items_text.split(",")]
	return [item for item in items if item]


def pairwise_agreement(gt_seq: list[str], pred_seq: list[str]) -> float:
	if len(gt_seq) != len(pred_seq) or not gt_seq:
		return 0.0
	index_pred = {item: idx for idx, item in enumerate(pred_seq)}
	agree = 0
	total = 0
	for i in range(len(gt_seq)):
		for j in range(i + 1, len(gt_seq)):
			total += 1
			item_i = gt_seq[i]
			item_j = gt_seq[j]
			if index_pred.get(item_i, -1) < index_pred.get(item_j, -1):
				agree += 1
	return agree / total if total else 0.0


def rotation_offset(gt_seq: list[str], pred_seq: list[str]) -> int | None:
	if len(gt_seq) != len(pred_seq):
		return None
	for k in range(len(gt_seq)):
		if pred_seq == gt_seq[k:] + gt_seq[:k]:
			return k
	return None


def is_adjacent_swap(gt_seq: list[str], pred_seq: list[str]) -> bool:
	if len(gt_seq) != len(pred_seq):
		return False
	diffs = [i for i, (a, b) in enumerate(zip(gt_seq, pred_seq)) if a != b]
	if len(diffs) != 2:
		return False
	i, j = diffs
	if j != i + 1:
		return False
	swapped = gt_seq[:]
	swapped[i], swapped[j] = swapped[j], swapped[i]
	return swapped == pred_seq


def is_any_two_swap(gt_seq: list[str], pred_seq: list[str]) -> bool:
	if len(gt_seq) != len(pred_seq):
		return False
	indices = [i for i, (a, b) in enumerate(zip(gt_seq, pred_seq)) if a != b]
	if len(indices) != 2:
		return False
	i, j = indices
	swapped = gt_seq[:]
	swapped[i], swapped[j] = swapped[j], swapped[i]
	return swapped == pred_seq


def is_two_adjacent_swaps(gt_seq: list[str], pred_seq: list[str]) -> bool:
	if len(gt_seq) != len(pred_seq):
		return False
	indices = [i for i, (a, b) in enumerate(zip(gt_seq, pred_seq)) if a != b]
	if len(indices) != 4:
		return False
	i1, i2, i3, i4 = indices
	if i2 != i1 + 1 or i4 != i3 + 1:
		return False
	if i3 <= i2:
		return False
	swapped = gt_seq[:]
	swapped[i1], swapped[i2] = swapped[i2], swapped[i1]
	swapped[i3], swapped[i4] = swapped[i4], swapped[i3]
	return swapped == pred_seq


def is_one_off_shift(gt_seq: list[str], pred_seq: list[str]) -> bool:
	if len(gt_seq) != len(pred_seq):
		return False
	index_pred = {item: idx for idx, item in enumerate(pred_seq)}
	for idx, item in enumerate(gt_seq):
		pred_idx = index_pred.get(item)
		if pred_idx is None:
			return False
		if abs(pred_idx - idx) > 1:
			return False
	return True


def error_type(gt_seq: list[str], pred_seq: list[str]) -> str:
	if pred_seq == gt_seq:
		return "correct"
	if pred_seq == list(reversed(gt_seq)):
		return "reverse"
	rotation = rotation_offset(gt_seq, pred_seq)
	if rotation is not None:
		return f"rotate_{rotation}"
	if is_adjacent_swap(gt_seq, pred_seq):
		return "adjacent_swap"
	if len(gt_seq) >= 2 and pred_seq:
		if pred_seq[0] == gt_seq[-1] and pred_seq[-1] == gt_seq[0]:
			if pred_seq[1:-1] == gt_seq[1:-1]:
				return "first_last_swap"
	if is_two_adjacent_swaps(gt_seq, pred_seq):
		return "two_adjacent_swaps"
	if is_any_two_swap(gt_seq, pred_seq):
		return "two_item_swap"
	if is_one_off_shift(gt_seq, pred_seq):
		return "one_off_shift"
	return "other"


def iter_json_files(folder: str) -> list[str]:
	if not os.path.isdir(folder):
		return []
	return sorted(
		os.path.join(folder, name)
		for name in os.listdir(folder)
		if name.endswith(".json")
	)


def load_json(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def summarize_group(records: list[dict]) -> dict:
	counts = {
		"total": len(records),
		"correct": sum(1 for r in records if r["is_correct"]),
		"incorrect": sum(1 for r in records if not r["is_correct"]),
	}
	accuracy = counts["correct"] / counts["total"] if counts["total"] else 0.0
	pos_acc = [r["position_accuracy"] for r in records if r["position_accuracy"] is not None]
	pair_agree = [r["pairwise_agreement"] for r in records if r["pairwise_agreement"] is not None]
	first_ok = [r["first_correct"] for r in records if r["first_correct"] is not None]
	last_ok = [r["last_correct"] for r in records if r["last_correct"] is not None]
	question_lengths = [r["question_length"] for r in records if r["question_length"] is not None]
	return {
		"counts": counts,
		"accuracy": accuracy,
		"avg_position_accuracy": mean(pos_acc) if pos_acc else None,
		"avg_pairwise_agreement": mean(pair_agree) if pair_agree else None,
		"first_correct_rate": mean(first_ok) if first_ok else None,
		"last_correct_rate": mean(last_ok) if last_ok else None,
		"question_length_avg": mean(question_lengths) if question_lengths else None,
		"question_length_median": median(question_lengths) if question_lengths else None,
	}


def build_item_position_counts(records: list[dict]) -> dict:
	position_counts = {
		"ground_truth": defaultdict(lambda: Counter()),
		"predicted": defaultdict(lambda: Counter()),
	}
	for record in records:
		for idx, item in enumerate(record.get("gt_seq", [])):
			position_counts["ground_truth"][item][idx + 1] += 1
		for idx, item in enumerate(record.get("pred_seq", [])):
			position_counts["predicted"][item][idx + 1] += 1
	return {
		side: {item: dict(counter) for item, counter in mapping.items()}
		for side, mapping in position_counts.items()
	}


def confusion_from_records(records: list[dict]) -> dict[str, dict[str, int]]:
	letters = ["A", "B", "C", "D"]
	confusion = {letter: Counter() for letter in letters}
	for record in records:
		gt_letter = record.get("gt_letter")
		pred_letter = record.get("pred_letter")
		if gt_letter in letters and pred_letter in letters:
			confusion[gt_letter][pred_letter] += 1
	return {gt: dict(counter) for gt, counter in confusion.items()}


def error_counts_from_records(records: list[dict]) -> dict[str, int]:
	counts = Counter()
	for record in records:
		counts[record.get("error_type")] += 1
	return dict(counts)


def write_confusion_plot(confusion: dict, output_path: str, title: str) -> None:
	import matplotlib.pyplot as plt
	letters = ["A", "B", "C", "D"]
	confusion_matrix = [
		[confusion.get(gt, {}).get(pred, 0) for pred in letters]
		for gt in letters
	]
	plt.figure(figsize=(6, 5))
	plt.imshow(confusion_matrix, cmap="Blues")
	plt.title(title)
	plt.xticks(range(len(letters)), letters)
	plt.yticks(range(len(letters)), letters)
	plt.xlabel("Predicted")
	plt.ylabel("Ground Truth")
	for i, row in enumerate(confusion_matrix):
		for j, value in enumerate(row):
			plt.text(j, i, str(value), ha="center", va="center", color="black")
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()


def write_bar_plot(values_map: dict, output_path: str, title: str, ylabel: str) -> None:
	import matplotlib.pyplot as plt
	labels = list(values_map.keys())
	values = [values_map[label] for label in labels]
	plt.figure(figsize=(8, 4))
	plt.bar(labels, values, color="#4C72B0")
	plt.title(title)
	plt.ylabel(ylabel)
	plt.xticks(rotation=30, ha="right")
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()


def write_hist_plot(values: list[float], output_path: str, title: str, xlabel: str) -> None:
	import matplotlib.pyplot as plt
	plt.figure(figsize=(6, 4))
	plt.hist(values, bins=10, color="#55A868", edgecolor="black")
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel("Count")
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()


def archive_existing_outputs(output_dir: str, archive_dir: str | None, patterns: list[str]) -> None:
	if not os.path.isdir(output_dir):
		return
	if not patterns:
		return
	stamp = time.strftime("%Y%m%d_%H%M%S")
	archive_root = archive_dir or os.path.join(output_dir, "archives", stamp)
	os.makedirs(archive_root, exist_ok=True)
	for name in os.listdir(output_dir):
		path = os.path.join(output_dir, name)
		if not os.path.isfile(path):
			continue
		if not any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
			continue
		dest = os.path.join(archive_root, name)
		shutil.move(path, dest)


def compute_word_position_bias(records: list[dict], source_key: str = "question_items") -> dict:
	position_counts = defaultdict(Counter)
	total_counts = Counter()
	for record in records:
		items = record.get(source_key) or []
		for idx, item in enumerate(items, start=1):
			word = str(item).lower()
			position_counts[idx][word] += 1
			total_counts[word] += 1

	positions = sorted(position_counts.keys())
	position_totals = {pos: sum(position_counts[pos].values()) for pos in positions}
	total_all = sum(total_counts.values())
	if total_all == 0:
		return {
			"positions": {},
			"total_counts": {},
		}

	results = {"positions": {}, "total_counts": dict(total_counts)}
	for pos in positions:
		pos_total = position_totals[pos]
		if pos_total == 0:
			continue
		bias_scores = {}
		for word, count in position_counts[pos].items():
			p_pos = count / pos_total
			p_global = total_counts[word] / total_all
			bias_scores[word] = p_pos / p_global if p_global > 0 else 0.0
		top_biased = sorted(bias_scores.items(), key=lambda x: x[1], reverse=True)[:10]
		top_words = []
		for word, bias in top_biased:
			top_words.append({
				"word": word,
				"bias_ratio": bias,
				"position_count": position_counts[pos][word],
				"global_count": total_counts[word],
			})
		results["positions"][str(pos)] = {
			"top_biased": top_words,
			"position_total": pos_total,
		}
	return results


def write_word_bias_plots(word_bias: dict, output_dir: str, suffix: str = "") -> None:
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ImportError:
		print("matplotlib not available; skipping word bias plots.")
		return

	positions = word_bias.get("positions", {})
	name_suffix = f"_{suffix}" if suffix else ""
	for pos, payload in positions.items():
		top_biased = payload.get("top_biased", [])
		if not top_biased:
			continue
		labels = [entry["word"] for entry in top_biased]
		values = [entry["bias_ratio"] for entry in top_biased]
		plt.figure(figsize=(8, 4))
		plt.bar(labels, values, color="#4C72B0")
		plt.title(f"Top 10 Biased Words at Position {pos}")
		plt.ylabel("Bias Ratio (position vs global)")
		plt.xticks(rotation=30, ha="right")
		plt.tight_layout()
		plt.savefig(os.path.join(output_dir, f"word_bias_position_{pos}{name_suffix}.png"))
		plt.close()


def write_plots(summary: dict, records: list[dict], output_dir: str) -> None:
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ImportError:
		print("matplotlib not available; skipping plot generation.")
		return

	correct_records = [r for r in records if r.get("is_correct")]
	incorrect_records = [r for r in records if not r.get("is_correct")]

	confusion_all = summary.get("confusion_matrix", {})
	confusion_correct = confusion_from_records(correct_records)
	confusion_incorrect = confusion_from_records(incorrect_records)
	write_confusion_plot(
		confusion_all,
		os.path.join(output_dir, "confusion_matrix_all.png"),
		"Confusion Matrix (All)",
	)
	write_confusion_plot(
		confusion_correct,
		os.path.join(output_dir, "confusion_matrix_correct.png"),
		"Confusion Matrix (Correct Only)",
	)
	write_confusion_plot(
		confusion_incorrect,
		os.path.join(output_dir, "confusion_matrix_incorrect.png"),
		"Confusion Matrix (Incorrect Only)",
	)

	error_counts_all = error_counts_from_records(records)
	error_counts_correct = error_counts_from_records(correct_records)
	error_counts_incorrect = error_counts_from_records(incorrect_records)
	if error_counts_all:
		write_bar_plot(
			error_counts_all,
			os.path.join(output_dir, "error_type_counts_all.png"),
			"Error Type Counts (All)",
			"Count",
		)
		write_bar_plot(
			error_counts_correct,
			os.path.join(output_dir, "error_type_counts_correct.png"),
			"Error Type Counts (Correct Only)",
			"Count",
		)
		write_bar_plot(
			error_counts_incorrect,
			os.path.join(output_dir, "error_type_counts_incorrect.png"),
			"Error Type Counts (Incorrect Only)",
			"Count",
		)

	position_acc_all = [r["position_accuracy"] for r in records if r["position_accuracy"] is not None]
	position_acc_correct = [r["position_accuracy"] for r in correct_records if r["position_accuracy"] is not None]
	position_acc_incorrect = [r["position_accuracy"] for r in incorrect_records if r["position_accuracy"] is not None]
	if position_acc_all:
		write_hist_plot(
			position_acc_all,
			os.path.join(output_dir, "position_accuracy_hist_all.png"),
			"Position Accuracy Distribution (All)",
			"Position Accuracy",
		)
		write_hist_plot(
			position_acc_correct,
			os.path.join(output_dir, "position_accuracy_hist_correct.png"),
			"Position Accuracy Distribution (Correct Only)",
			"Position Accuracy",
		)
		write_hist_plot(
			position_acc_incorrect,
			os.path.join(output_dir, "position_accuracy_hist_incorrect.png"),
			"Position Accuracy Distribution (Incorrect Only)",
			"Position Accuracy",
		)

	pair_agree_all = [r["pairwise_agreement"] for r in records if r["pairwise_agreement"] is not None]
	pair_agree_correct = [r["pairwise_agreement"] for r in correct_records if r["pairwise_agreement"] is not None]
	pair_agree_incorrect = [r["pairwise_agreement"] for r in incorrect_records if r["pairwise_agreement"] is not None]
	if pair_agree_all:
		write_hist_plot(
			pair_agree_all,
			os.path.join(output_dir, "pairwise_agreement_hist_all.png"),
			"Pairwise Agreement Distribution (All)",
			"Pairwise Agreement",
		)
		write_hist_plot(
			pair_agree_correct,
			os.path.join(output_dir, "pairwise_agreement_hist_correct.png"),
			"Pairwise Agreement Distribution (Correct Only)",
			"Pairwise Agreement",
		)
		write_hist_plot(
			pair_agree_incorrect,
			os.path.join(output_dir, "pairwise_agreement_hist_incorrect.png"),
			"Pairwise Agreement Distribution (Incorrect Only)",
			"Pairwise Agreement",
		)

	question_lengths_all = [r["question_length"] for r in records if r["question_length"] is not None]
	question_lengths_correct = [r["question_length"] for r in correct_records if r["question_length"] is not None]
	question_lengths_incorrect = [r["question_length"] for r in incorrect_records if r["question_length"] is not None]
	if question_lengths_all:
		write_hist_plot(
			question_lengths_all,
			os.path.join(output_dir, "question_length_hist_all.png"),
			"Question Length Distribution (All)",
			"Question Length (tokens)",
		)
		write_hist_plot(
			question_lengths_correct,
			os.path.join(output_dir, "question_length_hist_correct.png"),
			"Question Length Distribution (Correct Only)",
			"Question Length (tokens)",
		)
		write_hist_plot(
			question_lengths_incorrect,
			os.path.join(output_dir, "question_length_hist_incorrect.png"),
			"Question Length Distribution (Incorrect Only)",
			"Question Length (tokens)",
		)


def copy_error_samples(
	records: list[dict],
	correct_dir: str,
	incorrect_dir: str,
	output_dir: str,
	include_correct: bool,
	patterns: list[str],
) -> None:
	if not output_dir:
		return
	os.makedirs(output_dir, exist_ok=True)

	for record in records:
		is_correct = bool(record.get("is_correct"))
		if is_correct and not include_correct:
			continue
		if not is_correct and record.get("error_type") is None:
			continue

		error_label = "correct" if is_correct else str(record.get("error_type") or "unknown")
		base = os.path.splitext(record.get("file", ""))[0]
		if not base:
			continue

		src_dir = correct_dir if is_correct else incorrect_dir
		if not os.path.isdir(src_dir):
			continue

		dest_dir = os.path.join(output_dir, error_label)
		os.makedirs(dest_dir, exist_ok=True)

		for name in os.listdir(src_dir):
			if not name.startswith(base):
				continue
			if not any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
				continue
			src_path = os.path.join(src_dir, name)
			dest_path = os.path.join(dest_dir, name)
			if os.path.isfile(src_path):
				shutil.copy2(src_path, dest_path)


def main() -> None:
	parser = argparse.ArgumentParser(description="Analyze order prediction artifacts.")
	parser.add_argument(
		"--input_dir",
		default=(
			"/home/ramanathan/VLM/lmms-eval/experiment_artifacts_VSI_long_Order/qwen3/by_task_type/obj_appearance_order"
		),
		help="Folder with correct/incorrect subfolders.",
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		help="Where to write summary files (defaults to input_dir).",
	)
	parser.add_argument(
		"--json_name",
		default="order_analysis_summary.json",
		help="Summary JSON filename.",
	)
	parser.add_argument(
		"--csv_name",
		default="order_analysis_rows.csv",
		help="Per-example CSV filename.",
	)
	parser.add_argument(
		"--no_plots",
		action="store_true",
		help="Disable matplotlib plot generation.",
	)
	parser.add_argument(
		"--error_copy_dir",
		default=None,
		help="Copy samples grouped by error type into this folder.",
	)
	parser.add_argument(
		"--error_copy_patterns",
		default="*.json,*.png",
		help="Comma-separated patterns to copy (matched within each sample prefix).",
	)
	parser.add_argument(
		"--error_copy_include_correct",
		action="store_true",
		help="Also copy correct samples into a 'correct' folder.",
	)
	parser.add_argument(
		"--archive_outputs",
		action="store_true",
		help="Archive existing reports/plots before writing new ones.",
	)
	parser.add_argument(
		"--archive_dir",
		default=None,
		help="Custom archive directory (defaults to output_dir/archives/<timestamp>).",
	)
	parser.add_argument(
		"--word_bias",
		action="store_true",
		help="Analyze word position bias from question items.",
	)
	args = parser.parse_args()

	input_dir = args.input_dir
	output_dir = args.output_dir or input_dir

	correct_dir = os.path.join(input_dir, "correct")
	incorrect_dir = os.path.join(input_dir, "incorrect")
	correct_files = iter_json_files(correct_dir)
	incorrect_files = iter_json_files(incorrect_dir)

	records = []
	confusion = defaultdict(Counter)
	error_counts = Counter()

	for is_correct, file_list in ((True, correct_files), (False, incorrect_files)):
		for path in file_list:
			data = load_json(path)
			gt_letter = normalize_letter(data.get("ground_truth"))
			pred_letter = parse_pred_letter(data.get("answer"))
			options = options_map(data.get("options_texts"))
			gt_seq = options.get(gt_letter, [])
			pred_seq = options.get(pred_letter, [])
			if gt_letter and pred_letter:
				confusion[gt_letter][pred_letter] += 1
			seq_len = len(gt_seq) if gt_seq else None
			pos_acc = None
			pair_agree = None
			first_ok = None
			last_ok = None
			if gt_seq and pred_seq and len(gt_seq) == len(pred_seq):
				pos_acc = sum(1 for i in range(len(gt_seq)) if gt_seq[i] == pred_seq[i]) / len(gt_seq)
				pair_agree = pairwise_agreement(gt_seq, pred_seq)
				first_ok = gt_seq[0] == pred_seq[0]
				last_ok = gt_seq[-1] == pred_seq[-1]
			error = error_type(gt_seq, pred_seq) if gt_seq and pred_seq else "missing"
			if not is_correct:
				error_counts[error] += 1

			question_length = None
			question_words = data.get("question_words")
			if isinstance(question_words, list):
				question_length = len(question_words)

			record = {
				"file": os.path.basename(path),
				"doc_id": data.get("doc_id"),
				"is_correct": is_correct,
				"gt_letter": gt_letter,
				"pred_letter": pred_letter,
				"sequence_length": seq_len,
				"position_accuracy": pos_acc,
				"pairwise_agreement": pair_agree,
				"first_correct": first_ok,
				"last_correct": last_ok,
				"error_type": error,
				"question_length": question_length,
				"gt_seq": gt_seq,
				"pred_seq": pred_seq,
				"question_items": parse_question_items(data.get("question_text")),
			}
			records.append(record)

	summary = {
		"input_dir": input_dir,
		"overall": summarize_group(records),
		"correct_only": summarize_group([r for r in records if r["is_correct"]]),
		"incorrect_only": summarize_group([r for r in records if not r["is_correct"]]),
		"confusion_matrix": {gt: dict(counter) for gt, counter in confusion.items()},
		"error_type_counts": dict(error_counts),
		"item_position_counts": build_item_position_counts(records),
	}

	os.makedirs(output_dir, exist_ok=True)
	if args.archive_outputs:
		archive_existing_outputs(
			output_dir,
			args.archive_dir,
			patterns=["*.json", "*.csv", "*.png", "*.md"],
		)
	summary_path = os.path.join(output_dir, args.json_name)
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	csv_path = os.path.join(output_dir, args.csv_name)
	with open(csv_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"file",
				"doc_id",
				"is_correct",
				"gt_letter",
				"pred_letter",
				"sequence_length",
				"position_accuracy",
				"pairwise_agreement",
				"first_correct",
				"last_correct",
				"error_type",
				"question_length",
			],
		)
		writer.writeheader()
		for record in records:
			writer.writerow({key: record.get(key) for key in writer.fieldnames})

	if not args.no_plots:
		write_plots(summary, records, output_dir)

	if args.word_bias:
		word_bias_all = compute_word_position_bias(records, "question_items")
		word_bias_path = os.path.join(output_dir, "word_position_bias.json")
		with open(word_bias_path, "w", encoding="utf-8") as f:
			json.dump(word_bias_all, f, indent=2)
		write_word_bias_plots(word_bias_all, output_dir, "all")

		word_bias_gt = compute_word_position_bias(records, "gt_seq")
		word_bias_gt_path = os.path.join(output_dir, "word_position_bias_gt.json")
		with open(word_bias_gt_path, "w", encoding="utf-8") as f:
			json.dump(word_bias_gt, f, indent=2)
		write_word_bias_plots(word_bias_gt, output_dir, "gt")

		word_bias_pred = compute_word_position_bias(records, "pred_seq")
		word_bias_pred_path = os.path.join(output_dir, "word_position_bias_pred.json")
		with open(word_bias_pred_path, "w", encoding="utf-8") as f:
			json.dump(word_bias_pred, f, indent=2)
		write_word_bias_plots(word_bias_pred, output_dir, "pred")

		correct_records = [r for r in records if r.get("is_correct")]
		incorrect_records = [r for r in records if not r.get("is_correct")]
		word_bias_correct = compute_word_position_bias(correct_records, "question_items")
		word_bias_correct_path = os.path.join(output_dir, "word_position_bias_correct.json")
		with open(word_bias_correct_path, "w", encoding="utf-8") as f:
			json.dump(word_bias_correct, f, indent=2)
		write_word_bias_plots(word_bias_correct, output_dir, "correct")

		word_bias_incorrect = compute_word_position_bias(incorrect_records, "question_items")
		word_bias_incorrect_path = os.path.join(output_dir, "word_position_bias_incorrect.json")
		with open(word_bias_incorrect_path, "w", encoding="utf-8") as f:
			json.dump(word_bias_incorrect, f, indent=2)
		write_word_bias_plots(word_bias_incorrect, output_dir, "incorrect")

	if args.error_copy_dir:
		patterns = [p.strip() for p in args.error_copy_patterns.split(",") if p.strip()]
		if not patterns:
			patterns = ["*.json", "*.png"]
		copy_error_samples(
			records=records,
			correct_dir=correct_dir,
			incorrect_dir=incorrect_dir,
			output_dir=args.error_copy_dir,
			include_correct=args.error_copy_include_correct,
			patterns=patterns,
		)


if __name__ == "__main__":
	main()
