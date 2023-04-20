import pandas as pd
import numpy as np
import sys
import json
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# dataset = pd.DataFrame({'review_text': ['apple banana orange', 'dog cat orange', 'red blue orange yellow']})

from preprocess import preprocess

dataset_folder = "/Users/nate/Downloads/balanced_datasets"

TRAIN_SETS = {
     'amazon': f"{dataset_folder}/amazon/text/amazon_train.csv.gz",
     'steam': f"{dataset_folder}/steam/text/steam_train.csv.gz",
     'yelp': f"{dataset_folder}/yelp/text/yelp_train.csv.gz",
     'twitter': f"{dataset_folder}/twitter/text/twitter_train.csv.gz",
     'reddit': f"{dataset_folder}/reddit/text/reddit_train.csv.gz",
}

def get_dataset(path):
    data = pd.read_csv(path, compression='gzip')
    return data

def tokenize_review(review):
    return review.split()

def word_count(tokenized_reviews):
    return sum(len(review_list) for review_list in tokenized_reviews)

def review_count(tokenized_reviews):
    return len(tokenized_reviews)

def average_review_word_count(tokenized_reviews):
    return word_count(tokenized_reviews) / review_count(tokenized_reviews)

def flatten_reviews(tokenized_reviews):
    return [word for sublist in tokenized_reviews for word in sublist]

def unique_word_count(tokenized_reviews):
    word_list = flatten_reviews(tokenized_reviews)
    return len(set(word_list))

def positive_word_count(tokenized_reviews):
    positive_words = []
    with open('positive-words.txt') as file:
        positive_words = [line.rstrip() for line in file]

    word_list = flatten_reviews(tokenized_reviews)
    count = 0
    for word in word_list:
        if word in positive_words:
            count += 1
    return count
    

def negative_word_count(tokenized_reviews):
    negative_words = []
    with open('negative-words.txt') as file:
        negative_words = [line.rstrip() for line in file]

    word_list = flatten_reviews(tokenized_reviews)
    count = 0
    for word in word_list:
        if word in negative_words:
            count += 1
    return count

def word_frequencies(tokenized_reviews, count):
    word_list = flatten_reviews(tokenized_reviews)
    word_counts = Counter(word_list)
    word_counts_sorted = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
    word_counts_truncated = {key: word_counts_sorted[key] for key in list(word_counts_sorted.keys())[:count]}
    return word_counts_truncated

# def word_tfidf_scores(reviews, count):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(reviews)
#     feature_names = vectorizer.get_feature_names_out()
#     tfidf_means = np.mean(tfidf_matrix, axis=0)
#     sorted_indices = np.argsort(tfidf_means)[::-1]
    
    # print()
    # # for index in sorted_indices:
    # #     print(feature_names[index], tfidf_means[0][index])
    
    # print(feature_names[0:10])
    # print(tfidf_means[0:10])
    # for word, score in zip(feature_names, tfidf_means):
    #     print(f"{word}: {score}")

def create_wordcloud(tokenized_reviews):
    word_list = flatten_reviews(tokenized_reviews)
    text = ' '.join(word_list)
    wordcloud = WordCloud().generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def print_dataset_metrics(dataset_x):
    tokenized_reviews = dataset_x['review_text'].apply(tokenize_review).tolist()

    print(f"Total reviews: {review_count(tokenized_reviews)}")
    print(f"Total words: {word_count(tokenized_reviews)}")
    print(f"Average words per review: {average_review_word_count(tokenized_reviews)}")
    print(f"Total unique words: {unique_word_count(tokenized_reviews)}")
    print(f"Total positive words: {positive_word_count(tokenized_reviews)}")
    print(f"Total negative words: {negative_word_count(tokenized_reviews)}")

    print(f"Word frequency dictionary: {json.dumps(word_frequencies(tokenized_reviews, 10), indent=4)}")
    create_wordcloud(tokenized_reviews)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        dataset_name = sys.argv[1]
        print_dataset_metrics(get_dataset(TRAIN_SETS[dataset_name]))
    else:
        print("Please provide 2 arguments")
