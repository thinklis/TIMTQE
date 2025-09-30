import os
import json
import csv
import logging
import argparse
from typing import List, Tuple
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from sklearn import preprocessing


# -----------------------------
# Metrics
# -----------------------------
def pearson_corr(preds: List[float], labels: List[float]) -> float:
    return pearsonr(preds, labels)[0]


def spearman_corr(preds: List[float], labels: List[float]) -> float:
    return spearmanr(preds, labels)[0]


def rmse(preds: List[float], labels: List[float]) -> float:
    return np.sqrt(((np.asarray(preds, dtype=np.float32) -
                     np.asarray(labels, dtype=np.float32)) ** 2).mean())


def z_score(arr_list: List[float]) -> List[float]:
    """Compute z-score normalization."""
    arr = np.array(arr_list, dtype=np.float32)
    return preprocessing.scale(arr).tolist()


# -----------------------------
# File IO
# -----------------------------
def read_test_file(path: str, index: str = "index") -> pd.DataFrame:
    """Read label TSV file and return dataframe."""
    indices, originals, translations, means, z_means = [], [], [], [], []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])
            means.append(row["mean"])
            if not row['z_mean']:
                print('z_mean is none')
                print(row[index])
            z_means.append(row["z_mean"])

    return pd.DataFrame({
        'index': indices,
        'original': originals,
        'translation': translations,
        'mean': means,
        'z_mean': z_means
    })


def save_list_to_file(values: List[float], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for v in values:
            f.write(f"{v:.6f}\n")


def save_metrics(path: str,
                 pearson: float,
                 spearman: float,
                 rmse_value: float,
                 mae: float,
                 error_count: int) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"pearson:{pearson}\n")
        f.write(f"spearman:{spearman}\n")
        f.write(f"rmse_value:{rmse_value}\n")
        f.write(f"mae:{mae}\n")
        f.write(f"response error: {error_count}\n")


# -----------------------------
# Core Evaluation
# -----------------------------
def evaluate_single_file(model_answer_path: str,
                         label_path: str,
                         save_dir: str,
                         language: str,
                         predict_field_name: str) -> None:
    """Evaluate predictions for a single language file."""
    os.makedirs(save_dir, exist_ok=True)

    # Save paths
    save_model_answer_path = os.path.join(save_dir, f"{os.path.basename(model_answer_path)}_model_answer.txt")
    save_model_answer_zmean_path = os.path.join(save_dir, f"{os.path.basename(model_answer_path)}_model_answer_zmean.txt")
    save_label_path = os.path.join(save_dir, f"{os.path.basename(model_answer_path)}_label.txt")
    save_metric_path = os.path.join(save_dir, f"{os.path.basename(model_answer_path)}_metric.txt")

    # Load labels
    index_field = "segid" if language == "ru-en" else "index"
    df = read_test_file(label_path, index=index_field)
    z_labels = list(map(float, df['z_mean']))

    save_list_to_file(z_labels, save_label_path)

    # Load model answers
    model_answers: List[float] = []
    error_count = 0
    with open(model_answer_path, 'r', encoding="utf-8") as f, \
            open(save_model_answer_path, 'w', encoding="utf-8") as fout:
        for line in tqdm(f, desc=f"Processing {os.path.basename(model_answer_path)}"):
            try:
                record = json.loads(line.strip())
                raw_answer = record[predict_field_name]  # model_answer
                output_score = float(raw_answer.split()[-1].rstrip('.'))
                fout.write(f"{output_score}\n")
                model_answers.append(output_score)
            except (KeyError, ValueError, IndexError, json.JSONDecodeError):
                fout.write("0\n")
                model_answers.append(0.0)
                error_count += 1

    # Normalize
    z_model_answers = z_score(model_answers)
    save_list_to_file(z_model_answers, save_model_answer_zmean_path)

    # Metrics
    pearson = pearson_corr(z_labels, z_model_answers)
    spearman = spearman_corr(z_labels, z_model_answers)
    rmse_value = rmse(z_labels, z_model_answers)
    mae = mean_absolute_error(z_labels, z_model_answers)

    save_metrics(save_metric_path, pearson, spearman, rmse_value, mae, error_count)

    logging.info(f"Finished evaluating {model_answer_path}, "
                 f"Pearson={pearson:.4f}, Spearman={spearman:.4f}, "
                 f"RMSE={rmse_value:.4f}, MAE={mae:.4f}, Errors={error_count}")


# -----------------------------
# Main
# -----------------------------
def main(args):

    predict_field_name = "model_answer"  # model_answer (llama), predict (llava-next, qwen2.5-VL) 
    os.makedirs(args.save_score_dir, exist_ok=True)
    exp_name = os.path.basename(os.path.normpath(args.model_answer_dir))

    exp_save_dir = os.path.join(args.save_score_dir, exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)

    files_list = os.listdir(args.model_answer_dir)

    for file_name in files_list:
        language = file_name.split('_')[1].split('.')[0]
        print(language)

        lang_name = f"HistMTQE_{language}_test.tsv"
        lang_path = os.path.join(args.label_dir, lang_name)

        model_answer_path = os.path.join(args.model_answer_dir, file_name)

        lang_save_dir = os.path.join(
            exp_save_dir,
            os.path.splitext(file_name)[0]
        )
        os.makedirs(lang_save_dir, exist_ok=True)

        evaluate_single_file(
            model_answer_path=model_answer_path,
            label_path=lang_path,
            save_dir=lang_save_dir,
            language=language,
            predict_field_name=predict_field_name
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against reference labels.")
    parser.add_argument("-m", "--model_answer_dir", type=str, required=True,
                        help="Directory containing model answer JSONL files.")
    parser.add_argument("-l", "--label_dir", type=str, required=True,
                        help="Directory containing reference label TSV files.")
    parser.add_argument("-o", "--save_score_dir", type=str, required=True,
                        help="Directory to save evaluation results.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    main(args)