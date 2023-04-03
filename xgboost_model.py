import xgboost as xgb

from sklearn.metrics import accuracy_score
from preprocess import preprocess_train_test_sets

TRAIN_PATH = "C:/data/datasets/yelp/yelp_train.csv.gz"
TEST_PATH = "C:/data/datasets/yelp/yelp_test.csv.gz"
VECTOR_ENCODING = "tf_idf"

def create_model():
    return xgb.XGBClassifier()

def train(model, X_train, y_train):
    model.fit(X_train, y_train)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    return accuracy_score(y_test, y_pred)

def run():

    X_train, X_test, y_train, y_test = preprocess_train_test_sets(TRAIN_PATH, TEST_PATH, VECTOR_ENCODING)

    model = create_model()
    train(model, X_train, y_train)
    accuracy = evaluate(model, X_test, y_test)

    print(accuracy)

if __name__ == '__main__':
    run()
