import pandas as pd
import xgboost as xgb
import sys

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import preprocess

TRAIN_SETS = [
    ('steam', "C:/data/datasets/steam/steam_train.csv.gz"),
    ('yelp', 'C:/data/datasets/yelp/yelp_train.csv.gz'),
    ('amazon', 'C:/data/datasets/amazon/amazon_train.csv.gz')
]

TEST_SETS = [
    ('steam', "C:/data/datasets/steam/steam_test.csv.gz"),
    ('yelp', 'C:/data/datasets/yelp/yelp_test.csv.gz'),
    ('amazon', 'C:/data/datasets/amazon/amazon_test.csv.gz')
]

MODELS = {
    'xgboost': xgb.XGBClassifier()
}

def get_tf_idf_transformer(data):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data)

    return vectorizer

def get_dataset(path):
    data = pd.read_csv(path, compression='gzip')
    data = preprocess(data)
    return data

def train(model, X_train, y_train):
    model.fit(X_train, y_train)

def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)

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

def run_pipeline(model_type, training_dataset):

    model = MODELS[model_type]
    train_set = get_dataset(TRAIN_SETS[0][1])

    vectorizer = get_tf_idf_transformer(train_set['review_text'])

    X_train = vectorizer.transform(train_set['review_text'])
    y_train = train_set['review_score']

    print(f"Training on {training_dataset} dataset\n")
    train(model, X_train, y_train)

    for s in TEST_SETS:
        print(f"\nTesting on {s[0]} dataset\n")

        test_set = get_dataset(s[1])

        X_test = vectorizer.transform(test_set['review_text'])
        y_test = test_set['review_score']

        evaluate(model, X_test, y_test)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        model_type = sys.argv[1]
        train_set_name = sys.argv[2]

        run_pipeline(model_type, train_set_name)
    else:
        print("Please provide 2 arguments")
