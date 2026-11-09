
"""
SMS Spam Detection Using Machine Learning and Data Mining
Run: python main.py

Trains on the UCI SMS Spam Collection when available.
Uses word and character TF-IDF features to improve recognition
of spelling, spacing, punctuation, and formatting variations.
"""

from pathlib import Path
import csv
import io
import re
import unicodedata
import urllib.request
import zipfile

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
SAMPLE_FILE = DATA_DIR / "sms_spam_sample.csv"
MODEL_FILE = MODEL_DIR / "spam_detection_model.joblib"

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)


def normalize_text(text):
    """Normalize common SMS formatting variations."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()

    # Replace links, email addresses, and phone numbers consistently.
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        " EMAILTOKEN ",
        text,
    )
    text = re.sub(
        r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)",
        " PHONETOKEN ",
        text,
    )

    # Normalize common number disguises without deleting all digits.
    text = text.translate(str.maketrans({
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
    }))

    # Make repeated letters less extreme:
    # "freeeee" becomes "freee", retaining useful character patterns.
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)

    # Keep letters, numbers, and useful punctuation.
    text = re.sub(r"[^\w\s!?$£€%*#@]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def download_uci_dataset():
    """Download and extract the UCI SMS Spam Collection."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading the UCI SMS Spam Collection...")
        request = urllib.request.Request(
            UCI_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        with urllib.request.urlopen(request, timeout=20) as response:
            archive_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next(
                (
                    name for name in archive.namelist()
                    if name.endswith("SMSSpamCollection")
                ),
                None,
            )

            if member is None:
                raise FileNotFoundError(
                    "SMSSpamCollection was not found in the ZIP archive."
                )

            DATA_FILE.write_bytes(archive.read(member))

        print(f"Dataset downloaded to: {DATA_FILE}")
        return True

    except Exception as exc:
        print(f"Dataset download failed: {exc}")
        return False


def load_dataset():
    """Load the full dataset or the bundled demonstration CSV."""
    if not DATA_FILE.exists():
        download_uci_dataset()

    messages, labels = [], []

    if DATA_FILE.exists():
        with DATA_FILE.open(
            "r", encoding="utf-8", errors="replace"
        ) as file:
            for line in file:
                line = line.rstrip("\r\n")

                if "\t" not in line:
                    continue

                label, message = line.split("\t", 1)
                label = label.strip().lower()
                message = message.strip()

                if label in {"ham", "spam"} and message:
                    labels.append(label)
                    messages.append(message)

        if len(messages) >= 100:
            print(f"Loaded {len(messages)} messages from UCI.")
            print(
                "Class counts:",
                {label: labels.count(label) for label in set(labels)},
            )
            return messages, labels

    print("Using bundled sample data.")
    print("Results from a small sample are for demonstration only.")

    messages, labels = [], []

    with SAMPLE_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            message = (row.get("message") or "").strip()
            label = (row.get("label") or "").strip().lower()

            if message and label in {"ham", "spam"}:
                messages.append(message)
                labels.append(label)

    if len(messages) < 10 or len(set(labels)) < 2:
        raise RuntimeError(
            "Not enough usable data. Check data/sms_spam_sample.csv."
        )

    print(f"Loaded {len(messages)} sample messages.")
    return messages, labels


def build_model():
    """Build a model that learns word and character patterns."""
    word_features = TfidfVectorizer(
        preprocessor=normalize_text,
        lowercase=False,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=100000,
        sublinear_tf=True,
    )

    character_features = TfidfVectorizer(
        preprocessor=normalize_text,
        lowercase=False,
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        max_features=150000,
        sublinear_tf=True,
    )

    features = FeatureUnion([
        ("words", word_features),
        ("characters", character_features),
    ])

    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def train_model(messages, labels):
    """Train, evaluate, and save the classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    model = build_model()
    print("\nTraining word and character features...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n" + "=" * 58)
    print("MODEL EVALUATION")
    print("=" * 58)
    print(f"Training messages : {len(X_train)}")
    print(f"Testing messages  : {len(X_test)}")
    print(f"Accuracy          : {accuracy_score(y_test, predictions):.4f}")
    print(
        "Spam precision    : "
        f"{precision_score(y_test, predictions, pos_label='spam', zero_division=0):.4f}"
    )
    print(
        "Spam recall       : "
        f"{recall_score(y_test, predictions, pos_label='spam', zero_division=0):.4f}"
    )
    print(
        "Spam F1-score     : "
        f"{f1_score(y_test, predictions, pos_label='spam', zero_division=0):.4f}"
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        zero_division=0,
    ))

    print("Confusion matrix (rows=actual, columns=predicted; ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved trained model: {MODEL_FILE}")

    return model


def classify_message(model, message):
    """Predict whether a new message is spam."""
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = list(model.named_steps["classifier"].classes_)
    spam_probability = probabilities[classes.index("spam")] * 100

    print("\n--- Prediction Result ---")
    print(f"Classification: {'SPAM' if prediction == 'spam' else 'HAM (likely legitimate)'}")
    print(f"Estimated spam probability: {spam_probability:.2f}%")
    print("This probability is an estimate, not a guarantee.")

    if spam_probability >= 70:
        print("Advice: Treat this message with caution.")
    elif spam_probability >= 40:
        print("Advice: Uncertain result; check the sender and links.")
    else:
        print("Advice: The model considers this message less likely to be spam.")


def main():
    print("=" * 58)
    print("SMS SPAM DETECTION - MACHINE LEARNING PROJECT")
    print("=" * 58)

    try:
        messages, labels = load_dataset()
        model = train_model(messages, labels)
    except Exception as exc:
        print(f"\nCould not prepare the model: {exc}")
        print("Check the dataset and installed Python packages.")
        return

    while True:
        print("\nMENU")
        print("1. Check an SMS message")
        print("2. Show project information")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            message = input("Enter the SMS message: ").strip()

            if not message:
                print("Please enter a non-empty message.")
                continue

            classify_message(model, message)

        elif choice == "2":
            print("\nPROJECT INFORMATION")
            print("Title      : SMS Spam Detection Using Machine Learning")
            print("Data mining: Text mining and feature extraction")
            print("Features   : Word and character TF-IDF")
            print("Classifier : Logistic Regression")
            print("Evaluation : Accuracy, precision, recall, F1, confusion matrix")
            print("Dataset    : UCI SMS Spam Collection or demo sample")

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
