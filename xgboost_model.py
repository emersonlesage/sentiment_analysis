import sys
import xgboost as xgb

from sklearn.metrics import accuracy_score
from preprocess import get_train_test_sets

def create_model():
    return xgb.XGBClassifier()

def train(model, X_train, y_train):
    model.fit(X_train, y_train)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    return accuracy_score(y_test, y_pred)

def run(name, path, vector_encoding):

    X_train, X_test, y_train, y_test = get_train_test_sets(name, path, vector_encoding)

    model = create_model()

    train(model, X_train, y_train)
    
    accuracy = evaluate(model, X_test, y_test)

    print(accuracy)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        name = sys.argv[1]
        path = sys.argv[2]
        vector_encoding = sys.argv[3]

        run(name, path, vector_encoding)
    else:
        print("Please provide two arguments")
