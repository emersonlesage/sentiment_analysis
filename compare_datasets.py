import json
import gzip
import pandas as pd
import xgboost as xgb
import numpy as np
import sys
from itertools import chain

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import jaccard_score
from pydist2.distance import pdist1, pdist2

from preprocess import preprocess

dataset_folder = "/Users/nate/Downloads/balanced_datasets"

DATASETS = ['yelp', 'steam', 'amazon', 'reddit', 'twitter']


def get_dataset(path, is_csv):
    if is_csv:
        data = pd.read_csv(path, compression='gzip')
    else:
        with open(path, 'r') as f:
            data = json.load(f)
    return data

def get_tf_idf_transformer(data):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data)

    return vectorizer

def tokenize_review(review):
    return review.split()

def print_cosine_similarity(encoded_x, encoded_y, threshold):
    sim_matrix = cosine_similarity(encoded_x, encoded_y)
    avg_cos_sim = sim_matrix.mean()
    max_cos_sim = sim_matrix.max()

    num_above_threshold = (sim_matrix > threshold).sum()

    print(f"Average cosine similarity: {avg_cos_sim}")
    print(f"Max cosine similarity: {max_cos_sim}")
    print(f"Number of comparisons with cosine similarity meeting threshold {threshold}: {num_above_threshold}")


def print_euclidean_distance(encoded_x, encoded_y, threshold):
    sim_matrix = euclidean_distances(encoded_x, encoded_y)
    avg_cos_sim = sim_matrix.mean()
    max_cos_sim = sim_matrix.max()

    num_above_threshold = (sim_matrix > threshold).sum()

    print(f"Average cosine similarity: {avg_cos_sim}")
    print(f"Max cosine similarity: {max_cos_sim}")
    print(f"Number of comparisons with cosine similarity meeting threshold {threshold}: {num_above_threshold}")


def print_metric_info(metric, encoded_x, encoded_y, threshold):
    if metric == 'cosine_similarity':
        matrix = cosine_similarity(encoded_x, encoded_y)
    elif metric == 'euclidean_distance':
        matrix = euclidean_distances(encoded_x, encoded_y)

    avg = matrix.mean()
    max = matrix.max()
    above_threshold_count = (matrix > threshold).sum()

    print(f"Average {metric}: {avg:.4f}")
    # print(f"Max {metric}: {max}")
    # print(f"Count above threshold {threshold}: {above_threshold_count}")



def print_jaccard_score(dataset_x, dataset_y):
    tokenized_x = dataset_x['review_text'].apply(tokenize_review).tolist()
    tokenized_y = dataset_y['review_text'].apply(tokenize_review).tolist()

    # Test
    # tokenized_x = [ ['apple', 'orange'], ['apple', 'pear', 'kiwi'], ['pear'] ]
    # tokenized_y = [ ['apple', 'banana'], ['strawberry', 'pear', 'banana', 'pear'], ['pear', 'kiwi'] ]

    x_len = len(tokenized_x)
    y_len = len(tokenized_y)
    jaccard_scores = np.zeros((x_len, y_len))
    
    for i in range(x_len):
        for j in range(y_len):
            set_x = set(tokenized_x[i])
            set_y = set(tokenized_y[j])
            intersection = set_x.intersection(set_y)
            union = set_x.union(set_y)
            score = len(intersection) / len(union)      
            jaccard_scores[i, j] = score
        if(i % 1000 == 0):
            print(f"Jaccard Computation: {(i / x_len * 100):.2f}% complete")

    avg_jaccard_score = np.mean(jaccard_scores)
    print(f"Average Jaccard Score: {avg_jaccard_score}")

def get_dataset_path(set_name, data_type, train_or_test):
    file_ending = ".csv.gz" if data_type == "text" else ".json.gz"
    return f"{dataset_folder}/{set_name}/{data_type}/{set_name}_{train_or_test}{file_ending}"


def run_analysis(train_name, vector_encoding):
    text_training = get_dataset(get_dataset_path(train_name, f"vectors/{vector_encoding}", "train"), False)
    
    for test_name in DATASETS:
        print(f"\nAnalyzing {train_name} with {test_name}\n")
        test_set = get_dataset(get_dataset_path(test_name, f"vectors/{vector_encoding}", "test"), False)
        print_metric_info('cosine_similarity', text_training['vectors'], test_set['vectors'], 0.5)
        print_metric_info('euclidean_distance', text_training['vectors'], test_set['vectors'], 0.5)



if __name__ == '__main__':
    if len(sys.argv) == 3:
        dataset_name_x = sys.argv[1]
        vector_encoding = sys.argv[2]

        run_analysis(dataset_name_x, vector_encoding)
    else:
        print("Please provide 2 arguments")