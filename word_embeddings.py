import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def tokenize(data):
    return [x.split() for x in data]

def flatten_reviews(tokenized_reviews):
    return [word for sublist in tokenized_reviews for word in sublist]

def word_frequencies(tokenized_reviews, count):
    word_list = flatten_reviews(tokenized_reviews)
    word_counts = Counter(word_list)
    word_counts_sorted = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
    word_counts_truncated = {key: word_counts_sorted[key] for key in list(word_counts_sorted.keys())[:count]}
    return word_counts_truncated

def get_text(path):
    df = pd.read_csv(path)
    
    return df['review_text']

def get_embs(path):
    data = get_text(path)
    
    frqs = list(word_frequencies(tokenize(data), 1000).keys())      # ============ CAN CHANGE TO TOP ANYTHING DOESNT HAVE TO BE 1000 ====================

    emds = []

    for f in frqs:
        if f in embeddings_index.keys():
            emds.append(embeddings_index[f])
            
    return emds

def print_cosine_similarity(encoded_x, encoded_y, threshold):
    sim_matrix = cosine_similarity(encoded_x, encoded_y)
    avg_cos_sim = sim_matrix.mean()
    max_cos_sim = sim_matrix.max()
    
    print(f"\tAverage cosine similarity: {avg_cos_sim}")

embeddings_index = {}
with open('C:/data/glove.42B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

DATASETS = ['steam', 'yelp', 'amazon', 'twitter', 'reddit']

for d in DATASETS:
    print(f"============={d}=============")
    x = get_embs("C:/data/text/steam_train.csv.gz")

    for ds in DATASETS:
        print("\t" + ds)
        y = get_embs(f"C:/data/text/{ds}_test.csv.gz")
        print_cosine_similarity(x, y, 0.5)