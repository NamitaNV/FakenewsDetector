# 📰 Fake News Detector

A machine learning web app that classifies a news article as **REAL** or **FAKE** using TF-IDF features and Logistic Regression, built with scikit-learn and Streamlit.

> **Note:** This project is a learning/demo project. The model learns writing-style and source patterns from a fixed dataset. It does **not** verify facts and should not be used as an independent fact-checker.

---

## Features

- Compares three text-classification setups on a held-out validation split:
  - Passive Aggressive Classifier + TF-IDF
  - Logistic Regression + TF-IDF
  - Logistic Regression + Count Vectorizer
- Trains and saves the final model (Logistic Regression + TF-IDF with unigrams and bigrams)
- Interactive Streamlit app where you enter a title, author and article text
- Shows the predicted label (REAL / FAKE) with a model confidence score

## Tech Stack

- Python 3
- pandas, scikit-learn, matplotlib
- Streamlit

## Dataset

[Kaggle Fake News dataset](https://www.kaggle.com/c/fake-news) with the files `train.csv`, `test.csv` and `submit.csv`.

| Column | Description |
| ------ | ----------- |
| `title` | Headline of the article |
| `author` | Author name |
| `text` | Article body |
| `label` | `1` = FAKE, `0` = REAL (training data only) |

The title, author and text are combined into a single `total` field before vectorization.

## Project Structure

```
.
├── train.py                 # Data loading, model comparison, final training
├── app.py                   # Streamlit web app
├── model.pkl                # Saved Logistic Regression model (generated)
├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer (generated)
├── requirements.txt
└── README.md
```

> Rename `train.py` and `app.py` to match your actual file names.

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

Run the training script and the app in the same environment, because pickled models can break across scikit-learn versions.

### 3. Add the dataset

Download the dataset from Kaggle and place the CSV files in `../input/fake-news/`, or set the `FAKE_NEWS_INPUT` environment variable to your folder:

```bash
# Linux / macOS
export FAKE_NEWS_INPUT=/path/to/fake-news

# Windows (PowerShell)
$env:FAKE_NEWS_INPUT="C:\path\to\fake-news"
```

### 4. Train the model

```bash
python train.py
```

This prints the accuracy of each model, shows a confusion matrix for each, and saves `model.pkl` and `tfidf_vectorizer.pkl` next to the script.

### 5. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## How It Works

1. **Preprocessing:** Missing values are filled with a space, and `title + author + text` are joined into one string.
2. **Vectorization:** `TfidfVectorizer` with up to 50,000 features and unigrams + bigrams.
3. **Classification:** `LogisticRegression` (`max_iter=1000`).
4. **Evaluation:** 80/20 train/validation split (`random_state=0`), accuracy and confusion matrix.
5. **Inference:** The app builds the same combined text from the form inputs, vectorizes it, and returns the label and the predicted class probability as confidence.

## Results

| Model | Accuracy |
| ----- | -------- |
| Passive Aggressive + TF-IDF | _fill in_ |
| Logistic Regression + TF-IDF | _fill in_ |
| Logistic Regression + Count Vectorizer | _fill in_ |

Run `train.py` and replace the placeholders with your own numbers.

## Limitations

- The dataset is a fixed snapshot from around 2016, so performance on current news is likely to be much lower.
- The model picks up on style, vocabulary and author/source patterns rather than checking facts.
- In this dataset the author field can be a strong shortcut feature, so predictions may change noticeably if the author is left blank.
- The confidence score reflects the model's certainty, not the probability that the article is actually true or false.

## Future Improvements

- Save a single scikit-learn `Pipeline` instead of two separate pickle files
- Tune regularization with cross-validation and calibrate probabilities
- Report precision, recall and F1 in addition to accuracy
- Evaluate on more recent and more diverse news sources
- Try transformer-based models such as BERT

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

**Namita Nandakumar Varier**
