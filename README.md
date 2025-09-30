<p align="center">
  <img src="imgs/TIMTQE_logo.png" alt="TIMTQE Logo" width="150"/>
</p>

# TIMTQE: Text Image Machine Translation Quality Estimation Benchmark

This repository provides the official code and resources for **TIMTQE**,  
a benchmark dataset and evaluation framework for translation quality estimation (QE) on text images,  
covering both **synthetic** (MLQE-PE) and **historical** (HistMTQE) settings.  

---

## 📂 Dataset

The dataset is publicly available on HuggingFace Datasets:  

👉 [https://huggingface.co/datasets/TIMTQE/TIMTQE](https://huggingface.co/datasets/TIMTQE/TIMTQE)  

It includes:
- **MLQE-PE** – large-scale synthetic subset with rendered text images.  
- **HistMTQE** – human-annotated historical document subset.  

For detailed structure and examples, please check the HuggingFace dataset page.  

---

## ⚙️ Evaluation

We provide an evaluation toolkit to assess the performance of quality estimation models on TIMTQE.  
The main script is [`evaluate.py`](evaluate.py), which compares model predictions against human-annotated quality scores.

### 📌 Features
- **Input Format**: Predictions should be stored in a JSON, CSV, or TSV file, containing at least:
  - `id` (unique identifier of the sample)
  - `prediction` (the model’s QE score for the translation, typically on a 0–100 scale)
  - `label` (the human-annotated reference score)

- **Normalization**: To ensure fair comparison across systems, the script applies **z-score normalization** to model predictions.

- **Metrics**: The following evaluation metrics are computed:
  - **Pearson correlation** – measures the linear relationship between predictions and human scores.
  - **Spearman correlation** – assesses rank-based consistency between predictions and labels.
  - **RMSE (Root Mean Squared Error)** – penalizes larger deviations between predictions and reference scores.
  - **MAE (Mean Absolute Error)** – captures the average absolute difference between predictions and labels.

### 🚀 Usage
```bash
python evaluate.py \
  --pred_file results/predictions.json \
  --ref_file data/histmtqe/test.json \
  --output_dir outputs/
