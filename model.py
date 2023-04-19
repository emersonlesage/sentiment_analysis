import analysis
import sys
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from preprocess import preprocess_train_test_sets

def create_model():
    return xgb.XGBClassifier()

def train(model, X_train, y_train):
    model.fit(X_train, y_train)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    return accuracy_score(y_test, y_pred)

def run(name, path, vector_encoding, model_name):

    X_train, X_test, y_train, y_test = preprocess_train_test_sets(name, path, vector_encoding)

    model = get_model(model_name)

    train(model, X_train, y_train)

    y_pred = model.predict(X_test)  
    analysis.evaluate_classification(y_test, y_pred)

def get_model(model_name):
    models = {
        'mnb': MultinomialNB(),
        'svm': SVC(),
        'xgb': xgb.XGBClassifier()
    }
    if model_name in models:
        return models[model_name]
    else:
        return None

if __name__ == '__main__':
    if len(sys.argv) == 5:
        name = sys.argv[1]
        path = sys.argv[2]
        vector_encoding = sys.argv[3]
        model_name = sys.argv[4]

        run(name, path, vector_encoding, model_name)
    else:
        print("Please provide 4 arguments")
