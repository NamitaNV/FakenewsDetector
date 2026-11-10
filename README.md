# 📰 Fake News Detector

An NLP-powered Streamlit app that classifies a news article as **REAL** or **FAKE**. It compares several classical text-classification pipelines, keeps the best one, and explains which terms pushed the prediction in either direction.

> **Note:** This is an educational project. The model learns writing style and vocabulary patterns from a fixed dataset — it does **not** verify facts and should never be used as an independent fact-checker.

---

## Features

- Trains and compares three pipelines on the same held-out split:
  - Logistic Regression + TF-IDF
  - Passive Aggressive Classifier + TF-IDF
  - Logistic Regression + Count Vectors
- Automatically selects the winner by F1 score (accuracy as tie-breaker)
- Saves the trained model, vectorizer and a metadata report (`model_meta.json`)
- **Detect** tab: paste a headline and/or article, get a REAL/FAKE verdict with confidence
- **Why this verdict?** explains the decision by showing the highest-weighted terms in your text
- **Performance** tab: accuracy, precision, recall, F1, confusion matrix and comparison table
- **Dataset** tab: browse REAL/FAKE articles and load a random sample into the detector
- Robust input handling: HTML tags, URLs and extra whitespace are stripped; very short inputs are flagged

## Tech Stack

- Python 3
- pandas, NumPy, scikit-learn, matplotlib
- Streamlit

## Dataset

The project uses the [Kaggle Fake and Real News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (`True.csv` and `Fake.csv`), placed next to `fake_news.py`.

| File | Label | Description |
| --- | --- | --- |
| `True.csv` | `0` = REAL | Reuters-style real news articles |
| `Fake.csv` | `1` = FAKE | Articles from fake-news sources |

Both files contain `title`, `text`, `subject` and `date` columns. Only `title` and `text` are used for training.

### What happens to the data

1. HTML tags, URLs and repeated whitespace are removed.
2. `title` and `text` are concatenated into a single `total` field.
3. Rows with no usable text are dropped.
4. Articles with conflicting labels and exact duplicates are removed to avoid leakage.

Current dataset snapshot:

| Step | Count |
| --- | --- |
| Rows loaded | 44,898 |
| Rows without text | 9 |
| Duplicate articles removed | 5,794 |
| Articles used for training | **39,095** (21,195 REAL / 17,900 FAKE) |

## Project Structure

```
.
├── fake_news.py              # Cleaning, training, evaluation and artifact export
├── app.py                    # Streamlit web app
├── model.pkl                 # Saved classifier (generated)
├── tfidf_vectorizer.pkl      # Saved vectorizer (generated; name kept for compatibility)
├── model_meta.json           # Metrics, dataset stats and training metadata (generated)
├── metrics/                  # Confusion-matrix images (generated)
├── requirements.txt
├── True.csv                  # Dataset (download separately)
├── Fake.csv                  # Dataset (download separately)
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/NamitaNV/FakenewsDetector.git
cd FakenewsDetector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Train and run the app in the **same environment** — pickled scikit-learn models can break across library versions.

### 3. Add the dataset

Download `True.csv` and `Fake.csv` from Kaggle and place them next to `fake_news.py`, or point the trainer at another folder with the `FAKE_NEWS_DATA_DIR` environment variable:

```powershell
# Windows (PowerShell)
$env:FAKE_NEWS_DATA_DIR="C:\path\to\dataset"
```

```bash
# Linux / macOS
export FAKE_NEWS_DATA_DIR=/path/to/dataset
```

### 4. Train the model

```bash
python fake_news.py
```

This prints metrics for all three pipelines, writes a confusion matrix for each into `metrics/`, refits the best pipeline on the full dataset, and saves the artifacts next to the script.

Optional arguments:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--data-dir` | script folder | Folder containing `True.csv` and `Fake.csv` |
| `--output-dir` | script folder | Where artifacts and figures are written |
| `--test-size` | `0.2` | Held-out split size (stratified) |
| `--max-features` | `50000` | Maximum vocabulary size |
| `--max-iter` | `1000` | Maximum solver iterations |
| `--c` | `1e5` | Logistic Regression inverse regularization strength |
| `--seed` | `0` | Random seed |

Example:

```bash
python fake_news.py --test-size 0.2 --max-features 50000 --seed 42
```

### 5. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## How It Works

### Training (`fake_news.py`)

1. **Load** `True.csv` (REAL) and `Fake.csv` (FAKE) and assign labels.
2. **Clean** HTML, URLs and whitespace; combine `title + text`.
3. **Deduplicate** exact articles and drop rows whose text carries conflicting labels.
4. **Split** the data 80/20, stratified by label, with a fixed random seed.
5. **Evaluate** three vectorizer/classifier combinations with accuracy, precision, recall, F1 and a confusion matrix.
6. **Select** the best pipeline by F1, refit it on the full dataset, and save it.

### Inference (`app.py`)

1. The entered headline and article are joined into one string (same shape as training).
2. The saved vectorizer converts it into numeric features.
3. The saved classifier predicts REAL or FAKE and returns per-class probabilities.
4. For linear models, each term's TF-IDF weight is multiplied by its learned coefficient to show which words pulled toward FAKE or REAL.

## Results

Held-out 20% validation split (stratified, `random_state=0`):

| Pipeline | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| **Logistic Regression + Count Vectors** | **99.59%** | **99.53%** | **99.58%** | **99.55%** |
| Passive Aggressive + TF-IDF | 99.48% | 99.55% | 99.30% | 99.43% |
| Logistic Regression + TF-IDF | 99.32% | 99.36% | 99.16% | 99.26% |

The current saved model is **Logistic Regression + Count Vectors**, refit on the full dataset. Metrics are regenerated every time you run `fake_news.py` and are also displayed in the app's **Performance** tab.

> High accuracy here reflects how separable this particular dataset is, not how well the model generalizes to today's news.

## Limitations

- The dataset is a fixed snapshot from around 2016, so real-world accuracy on current articles will be much lower.
- The model learns style, vocabulary and source-specific patterns, not factual correctness.
- Reuters articles in `True.csv` share a distinctive style, which makes the classes easier to separate than real-world news.
- Count vectors and TF-IDF give equal weight to terms regardless of context, so sarcasm, quotes and negations are easily missed.
- The confidence score is the model's certainty, **not** the probability that an article is true.
- Inputs under ~15 words trigger a warning because short text is unreliable.

## Future Improvements

- Save one scikit-learn `Pipeline` instead of separate model/vectorizer pickles
- Tune regularization with cross-validation and calibrate probabilities
- Evaluate on recent, more diverse news sources to measure real-world drift
- Add cross-validation instead of a single train/test split
- Experiment with transformer models such as DistilBERT or RoBERTa
- Add tests for preprocessing, training and inference

## Author

**Namita Nandakumar Varier**

- GitHub: [@NamitaNV](https://github.com/NamitaNV)
- Repository: [FakenewsDetector](https://github.com/NamitaNV/FakenewsDetector)
