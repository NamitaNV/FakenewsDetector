import os
import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

INPUT_DIR = os.environ.get("FAKE_NEWS_INPUT", os.path.join("..", "input", "fake-news"))


def list_inputs(input_dir):
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def load_data(input_dir):
    train = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test = pd.read_csv(os.path.join(input_dir, "test.csv"))
    submit = pd.read_csv(os.path.join(input_dir, "submit.csv"))

    print(f"Train Shape : {train.shape}")
    print(f"Test Shape : {test.shape}")
    print(f"Submit Shape : {submit.shape}")

    train.info()
    print(train.isnull().sum())
    print(train.dtypes.value_counts())

    train = train.fillna(" ")
    test = test.fillna(" ")

    train["total"] = train["title"] + " " + train["author"] + " " + train["text"]
    test["total"] = test["title"] + " " + test["author"] + " " + test["text"]

    return train, test, submit


def plot_confusion_matrix(cm, classes, title):
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot()
    plt.title(title)
    plt.show()


def evaluate(name, classifier, vectorizer, X_train, y_train, X_test, y_test):
    train_matrix = vectorizer.fit_transform(X_train)
    test_matrix = vectorizer.transform(X_test)

    classifier.fit(train_matrix, y_train)
    predictions = classifier.predict(test_matrix)

    accuracy = metrics.accuracy_score(y_test, predictions)
    print(f"{name}: accuracy = {accuracy:.4f}")

    cm = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    plot_confusion_matrix(cm, classes=["TRUE", "FAKE"], title=f"Confusion matrix for {name}")
    return accuracy


def train_and_save_model(train):
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    train_matrix = vectorizer.fit_transform(train["total"])

    model = LogisticRegression(C=1e5, max_iter=1000, random_state=0)
    model.fit(train_matrix, train["label"])

    output_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(output_dir, "model.pkl"), "wb") as file:
        pickle.dump(model, file)

    with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), "wb") as file:
        pickle.dump(vectorizer, file)

    print("Model and vectorizer saved successfully!")


def main():
    list_inputs(INPUT_DIR)
    train, test, submit = load_data(INPUT_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        train["total"], train.label, test_size=0.20, random_state=0
    )

    evaluate(
        "a PA Classifier with Tf-IDF",
        PassiveAggressiveClassifier(max_iter=1000, random_state=0),
        TfidfVectorizer(max_features=50000, ngram_range=(1, 2)),
        X_train, y_train, X_test, y_test,
    )
    evaluate(
        "a Logistic Regression with Tf-IDF",
        LogisticRegression(C=1e5, max_iter=1000),
        TfidfVectorizer(max_features=50000, ngram_range=(1, 2)),
        X_train, y_train, X_test, y_test,
    )
    evaluate(
        "a Logistic Regression with Count Vectorizer",
        LogisticRegression(C=1e5, max_iter=1000),
        CountVectorizer(max_features=50000, ngram_range=(1, 2)),
        X_train, y_train, X_test, y_test,
    )

    train_and_save_model(train)


if __name__ == "__main__":
    main()
