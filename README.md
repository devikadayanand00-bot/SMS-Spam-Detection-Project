#SMS Spam Detection Using Machine Learning and Data Mining

## Overview
A machine-learning/data-mining project that classifies an SMS as SPAM or HAM (legitimate).
It uses TF-IDF text features and a Multinomial Naive Bayes classifier.

## Folder structure
SMS_Spam_Detection_Project/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── sms_spam_sample.csv
└── models/
    └── (trained model is saved here after the first run)

## Requirements
- Python 3.10 or newer recommended
- PyCharm
- Internet connection for automatic download of the full UCI dataset (recommended)

## Setup in PyCharm
1. Extract the ZIP file.
2. Open PyCharm → Open → choose the extracted `SMS_Spam_Detection_Project` folder.
3. Create/select a Python interpreter (Python 3.10+ recommended).
4. Open the PyCharm Terminal at the bottom.
5. Run:
   `python -m pip install -r requirements.txt`
6. Right-click `main.py` → Run 'main'.

## Dataset
The program tries to download the UCI SMS Spam Collection dataset automatically.
If that fails, it uses `data/sms_spam_sample.csv` so the program can still be tested.
The bundled sample is small and is only for demonstrating the workflow; do not use its metrics as final project results.
For a proper report, run with the full UCI dataset and record the metrics printed by the program.

## How to use
- Choose `1` to enter an SMS and receive a prediction.
- Choose `2` to view project information.
- Choose `3` to exit.

## Data-mining / ML workflow
1. Collect labelled SMS data.
2. Preprocess text by lowercasing, accent normalization, and tokenization performed by the vectorizer.
3. Convert text into numeric TF-IDF features using unigrams and bigrams.
4. Split the dataset into training and testing sets (80/20).
5. Train Multinomial Naive Bayes.
6. Evaluate with accuracy, spam precision, spam recall, F1-score, classification report, and confusion matrix.
7. Predict the class of new SMS messages.

## Important notes
- Spam probability is an estimate, not a guarantee.
- Do not enter private or sensitive messages when demonstrating the application.
- For a final academic submission, cite the dataset source and explain the train/test split.
