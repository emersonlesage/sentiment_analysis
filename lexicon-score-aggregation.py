# Baseline method (not very good)
import analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

dataset_folder = "/Users/nate/Downloads/datasets"

TEST_SETS = [
    ('steam', f"{dataset_folder}/steam_test.csv.gz"),
    ('yelp', f"{dataset_folder}/yelp_test.csv.gz"),
    ('amazon', f"{dataset_folder}/amazon_test.csv.gz")
]

df = pd.read_csv(TEST_SETS[2][1])

positive_words = []
with open('positive-words.txt') as file:
    positive_words = [line.rstrip() for line in file]

negative_words = []
with open('negative-words.txt') as file:
    negative_words = [line.rstrip() for line in file]

scores = []
TP = 0
TN = 0
FP = 0
FN = 0

y_pred = []
y_test = []

for index, row in df.iterrows():
    review_text = str(row[0])
    score = 0
    for word in review_text.split():
        if word in positive_words:
            score += 1
        if word in negative_words:
            score -= 1
    if score == 0:
        score = random.randint(-1, 0)
    if score >= 0:
        if row[1] == 1:
            TP += 1
        else:
            FP += 1
    else:
        if row[1] == 0:
            TN += 1
        else:
            FN += 1
    # scores.append(score)
    y_pred.append(1 if score >= 0 else 0)
    y_test.append(row[1])
    print(f"Finished reading {index} with score: {score}")

y_pred = np.array(y_pred)
y_test = np.array(y_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
