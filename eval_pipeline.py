import pandas as pd
import xgboost as xgb
import sys
import json

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import MultinomialNB

MODELS = {
    'xgboost': xgb.XGBClassifier(),
    'bayes': MultinomialNB()
}

DATASETS = ['yelp', 'steam', 'amazon', 'reddit', 'twitter']

def train(model, X_train, y_train):
    model.fit(X_train, y_train)

def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)

    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    print(f"AUC-ROC Score: {auc_roc:.4f}")

def get_vectors_labels(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data['vectors'], data['labels']

def run_pipeline(model_type, training_dataset, vector_encoding):

    train_path = f"C:/data/vectors/{vector_encoding}/{training_dataset}_train.json.gz"
    model = MODELS[model_type]

    print(f"Loading {training_dataset} dataset")
    X_train, y_train = get_vectors_labels(train_path)

    print(f"Training on {training_dataset} dataset\n")
    train(model, X_train, y_train)

    test_sets = [f"C:/data/vectors/{vector_encoding}/{i}_test.json.gz" for i in DATASETS]

    for s in test_sets:
        print(f"\nTesting on {s}\n")

        X_test, y_test = get_vectors_labels(s)

        evaluate(model, X_test, y_test)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        model_type = sys.argv[1]
        train_set_name = sys.argv[2]
        vector_encoding = sys.argv[3]

        run_pipeline(model_type, train_set_name, vector_encoding)
    else:
        print("Please provide 2 arguments")
