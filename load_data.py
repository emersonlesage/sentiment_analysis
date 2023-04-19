import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

STEAM_PATH = 'C:/data/raw/steam_raw.csv.gz'
YELP_PATH = 'C:/data/raw/yelp_raw.csv.gz'
AMAZON_PATH = 'C:/data/raw/amazon_automotive_raw.csv.gz'
TWITTER_PATH = 'C:/data/raw/twitter_raw.csv.gz'
REDDIT_PATH = 'C:/data/raw/reddit_raw.csv.gz'

PATHS = {
    'steam': STEAM_PATH,
    'yelp': YELP_PATH,
    'amazon': AMAZON_PATH,
    'twitter': TWITTER_PATH,
    'reddit': REDDIT_PATH
}

def batch_load(path):
    # load file in batches, take text and scores

    data = []

    for chunk in pd.read_csv(path, compression='gzip', chunksize=100000):

        # Process each chunk of data here
        data.append(chunk[['review_text', 'review_score']])

    df = pd.concat(data, ignore_index=True)
    df.columns = ['review_text', 'review_score']
    df = df.dropna()

    return df

def sample(df):
    # take a 1% stratified sample of raw data

    sample_df, _ = train_test_split(
        df,
        train_size=0.01,
        stratify=df['review_score'],
        random_state=69
    )

    return sample_df


def load(name):

    print(f"\nLoading {name} dataset...", end='')
    data = batch_load(PATHS[name])

    data['review_text'] = data['review_text'].astype('U').values

    print(f" DONE: {len(data)} records")
    if name == 'steam' or name == 'reddit':
        # replace -1 with 0
        data.loc[:, "review_score"] = (
            data["review_score"]
            .apply(lambda x: 0 if x == -1 else x)
        )
    elif name == 'twitter':
        data['review_score'] = data['review_score'].replace({4: 1})
    else:
        # remove 3 start reviews (neutral sentiment)
        data = data[data['review_score'] != 3]

        # map 1 and 2 start reviews to a negative sentiment (0)
        # map 4 and 5 start reviews to a positive sentiment (1)
        data['review_score'] = data['review_score'].replace({1: 0, 2: 0, 4: 1, 5: 1})

    if name != 'twitter':
        print("\nBalancing dataset ...", end='')
        data = balance(data)
        print(f" DONE: {len(data)} records")

    if name != 'reddit':
        print("\nSampling dataset ...", end='')
        data = sample(data)
        print(f" DONE {len(data)} records")

    return data

def balance(data):
    # balance the number of postitve and negative reviews by undersampling the 
    # majority class

    # Separate the majority and minority classes
    X_majority = data[data['review_score'] == 1]['review_text']
    X_minority = data[data['review_score'] == 0]['review_text']

    # Compute the number of samples in the minority class
    n_minority = len(X_minority)

    # Compute the number of samples in the majority class
    n_majority = len(X_majority)

    # Determine the size of the subset to sample from the majority class
    # This ensures that we don't try to sample more elements than the size of the array
    n_majority_subset = min(n_majority, n_minority)

    # Undersample the majority class to have the same number of samples as the minority class
    X_majority_undersampled = resample(X_majority, n_samples=n_majority_subset, replace=False, random_state=42)

    # Combine the undersampled majority class and the original minority class to create the balanced dataset
    X_balanced = pd.concat([X_majority_undersampled, X_minority])
    y_balanced = np.concatenate((np.ones(n_minority), np.zeros(n_minority)))

    balanced_data = pd.DataFrame(X_balanced).reset_index(drop=True)
    print(len(balanced_data))
    balanced_data['review_score'] = y_balanced

    return balanced_data

def split(data):

    print("\nPerforming an 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        data["review_text"],
        data["review_score"],
        stratify=data["review_score"],
        test_size=0.2,
        random_state=69,
    )

    train = pd.DataFrame()
    test = pd.DataFrame()

    train["review_text"] = X_train
    train["review_score"] = y_train

    test["review_text"] = X_test
    test["review_score"] = y_test

    print(f"\ttrain: {len(train)} records")
    print(f"\ttest: {len(test)} records")

    return train, test

def save_train_test(dataset, out_path):

    print("Loading dataset...")
    data = load(
        dataset
    )

    train, test = split(data)

    train_path = f"{out_path}/{dataset}_train.csv.gz"
    test_path = f"{out_path}/{dataset}_test.csv.gz"

    print("Writing dataset to disk...")
    train.to_csv(train_path, compression='gzip', index=False)
    print(f"Train set saved to {train_path}")

    print("Writing dataset to disk...")
    test.to_csv(test_path, compression='gzip', index=False)
    print(f"Test set saved to {test_path}")


if __name__ == '__main__':
    if len(sys.argv) == 3:

        save_train_test(sys.argv[1], sys.argv[2])
    else:
        print("Please provide two arguments")