import sys
import json

import pandas as pd

from load_data import load, split
from preprocess import preprocess, vectorize


def save_preprocessed_data(dataset):

    text_path = f"C:/data/4710/{dataset}/text/{dataset}"   # path for the preprocess text data
    tf_idf_path = f"C:/data/4710/{dataset}/tf_idf/{dataset}" 
    s_embedding_path = f"C:/data/4710/{dataset}/sentence_384/{dataset}" 

    # load raw data
    data = load(
        dataset
    )

    data = preprocess(data)
    train, test = split(data)               # train and test sets

    # get a contiguous index
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    write(train, test, text_path)

    train_v, test_v = vectorize(train, test, "tf_idf")
    write_json(train_v, test_v, tf_idf_path)

    train_v, test_v = vectorize(train, test, "sentence")
    write_json(train_v, test_v, s_embedding_path)


def write_json(train, test, path):
    train_path = path+"_train.json.gz"
    test_path = path+"_test.json.gz"

    print(f"\tWriting train file to {train_path}")
    with open(train_path, "w") as f:
        json.dump(train, f)

    print(f"\tWriting test file to {test_path}")
    with open(test_path, "w") as f:
        json.dump(test, f)

def write(train, test, path):

    train_path = path+"_train.csv.gz"
    test_path = path+"_test.csv.gz"

    print(f"Writing train file to {train_path}")
    train.to_csv(train_path, compression='gzip', index=False)

    print(f"Writing test file to {test_path}")
    test.to_csv(test_path, compression='gzip', index=False)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        save_preprocessed_data(sys.argv[1])
    else:
        print("Please provide 1 argument")

