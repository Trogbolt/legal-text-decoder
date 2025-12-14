# Legal Text Decoder — Deep Learning Project (VITMMA19)

## Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Kozits Patrik
- **Aiming for +1 Mark**: No

## Problem Statement

The goal of this project is to build an NLP model that predicts how easy or difficult a paragraph from Hungarian legal documents (Terms and Conditions / General Terms of Use) is for an average user to understand.

The model predicts a score on a **1–5** scale:

1. **Very hard / unclear**  
2. **Hard**  
3. **Medium (requires strong focus)**  
4. **Understandable after reading**  
5. **Easy / immediately understandable**

## Solution Overview

This repository implements a full ML pipeline inside Docker:

1. **Data download (optional)**: download and unpack the dataset ZIP if `/data/raw` is empty.
2. **Data preprocessing**: parse Label Studio JSON exports, extract `(text, label)` pairs, and create `train/val/test` CSV files.
3. **Baselines**:
   - A simple heuristic baseline based on **word count quantiles**
   - A classical ML baseline: **TF-IDF + Logistic Regression**
4. **Model training**:
   - Fine-tune **`bert-base-multilingual-cased`** with a classification head (5 classes)
   - Early stopping using validation **F1-weighted**
   - Save `model_best` and `model_last`
5. **Evaluation**: evaluate `model_best` on the test set; log metrics, classification report, confusion matrix
6. **Inference demo**: run the trained model on a few example texts and log predicted label + confidence + probabilities

## Data Acquisition and Preparation

### Raw dataset format

The raw dataset is a Label Studio export. The folder structure under `/data/raw` is expected to contain annotator directories, each containing JSON exports.

Ignored directories:
- `consensus`
- `sample`

Some files that do not meet the requirements may be present; these are skipped automatically if not in Label Studio task-list format.

### Automated preparation

The preprocessing script:

- Detects the ZIP internal root directory (if the extracted ZIP contains one top-level folder).
- Iterates through annotator folders.
- For each JSON file:
  - Loads Label Studio tasks (must be a list)
  - Extracts `task["data"]["text"]`
  - Extracts the label from `annotations[0]["result"][0]["value"]["choices"][0]`
  - Converts the label to integer **1–5**
- Creates:
  - `/app/output/train.csv`
  - `/app/output/val.csv`
  - `/app/output/test.csv`

Splitting:
- 80% train, 10% validation, 10% test
- Stratified by label (`random_state=42`)

### Final dataset sizes

- Total labeled examples: **3744**
- Train / Val / Test: **2995 / 374 / 375**

## Configuration

All main hyperparameters and paths are stored in `src/config.py`.

Current configuration:

- **Model**
  - `MODEL_NAME = bert-base-multilingual-cased`
  - `NUM_LABELS = 5`
  - `MAX_LENGTH = 256`

- **Training**
  - `EPOCHS = 8`
  - `TRAIN_BATCH_SIZE = 4`
  - `EVAL_BATCH_SIZE = 8`
  - `LEARNING_RATE = 2e-5`
  - `WEIGHT_DECAY = 0.01`
  - `EARLY_STOPPING_PATIENCE = 2`

- **Paths**
  - `PATH_TRAIN = /app/output/train.csv`
  - `PATH_VAL   = /app/output/val.csv`
  - `PATH_TEST  = /app/output/test.csv`
  - `MODEL_BEST_DIR = /app/output/model_best`
  - `MODEL_LAST_DIR = /app/output/model_last`

## Results

### Baseline 1 — Word-count heuristic (quantile thresholds)

- Thresholds: 19.0, 31.0, 46.0, 70.0
- **Accuracy:** 0.2987  
- **F1-weighted:** 0.3148  

This simple heuristic baseline provides a weak reference point and demonstrates that word count alone is insufficient for modeling legal text understandability.

---

### Baseline 2 — TF-IDF + Logistic Regression

- **Accuracy:** 0.4347  
- **F1-weighted:** 0.4099  

The TF-IDF baseline significantly outperforms the heuristic approach and serves as a strong classical machine learning reference.

---

### Fine-tuned Transformer — BERT (`bert-base-multilingual-cased`)

**Training device:**
- CUDA GPU: **NVIDIA GeForce GTX 1650**

**Model size:**
- Total parameters: **177,857,285**
- Trainable parameters: **177,857,285**
- Frozen parameters: **0**

The multilingual BERT model was fine-tuned end-to-end for a 5-class text classification task.  
Early stopping was applied based on validation **F1-weighted** score.

**Best validation result (`model_best`):**
- Best observed **val F1-weighted:** **0.4521** (epoch 3)

**Final test evaluation (`model_best`):**
- **Test accuracy:** **0.4640**  
- **Test F1-weighted:** **0.4573**  

A full classification report and confusion matrix are available in `log/run.log`.

---

### Quick comparison

| Model | Accuracy | F1-weighted |
|------|----------:|------------:|
| Word-count heuristic | 0.2987 | 0.3148 |
| TF-IDF + Logistic Regression | 0.4347 | 0.4099 |
| **Fine-tuned BERT** | **0.4640** | **0.4573** |

## Docker Instructions

### Build

Build Image:

```bash
docker build -t legal-text-decoder:1.0 .

```

### Run

Run the container:

```bash
docker run --rm --gpus all -v "${PWD}/data:/data" -v "${PWD}/output:/app/output" legal-text-decoder:1.0 > log/run.log 2>&1

```

This command executes the full pipeline:

- Data preprocessing
- Baseline models
- Model training
- Evaluation
- Inference

All logs are written to log/run.log.

### Project structure

```
legal-text-decoder/
├── data/
│ └── raw/ # Raw Label Studio dataset exports (JSON files)
│
├── log/
│ └── run.log # Full pipeline execution log (training, evaluation, inference)
│
├── notebook/
│ ├── 01-data-exploration.ipynb # Exploratory data analysis (EDA)
│ └── 02-label-analysis.ipynb # Label distribution and class balance analysis
│
├── src/
│ ├── 00_download_and_unpack.py # Download and extract dataset ZIP if raw data is missing
│ ├── 01_data_preprocessing.py # Parse Label Studio JSON files and create train/val/test CSVs
│ ├── baseline_heuristic.py # Heuristic baseline using word-count thresholds
│ ├── baseline.py # TF-IDF + Logistic Regression baseline model
│ ├── 02_train.py # Fine-tuning BERT model with early stopping
│ ├── 03_evaluation.py # Evaluation on test set (accuracy, F1, confusion matrix)
│ ├── 04_inference.py # Inference demo on new example texts
│ ├── config.py # Central configuration (paths, hyperparameters)
│ └── utils.py # Logging utilities
│
├── Dockerfile # Docker image definition
├── requirements.txt # Python dependencies with fixed versions
├── run.sh # Script to execute the full pipeline inside the container
└── README.md # Project documentation
```