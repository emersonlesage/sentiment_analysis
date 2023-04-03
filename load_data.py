import pandas as pd
import sys
from sklearn.model_selection import train_test_split

STEAM_PATH = 'C:/data/raw/steam_raw.csv.gz'
YELP_PATH = 'C:/data/raw/yelp_raw.csv.gz'
AMAZON_PATH = 'C:/data/raw/amazon_automotive_raw.csv.gz'

PATHS = {
    'steam': STEAM_PATH,
    'yelp': YELP_PATH,
    'amazon': AMAZON_PATH
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

    data = batch_load(PATHS[name])
    data = sample(data)

    if name == 'steam':
        # replace -1 with 0
        data.loc[:, "review_score"] = (
            data["review_score"]
            .apply(lambda x: 0 if x == -1 else x)
        )
    else:
        # remove 3 start reviews (neutral sentiment)
        data = data[data['review_score'] != 3]

        # map 1 and 2 start reviews to a negative sentiment (0)
        # map 4 and 5 start reviews to a positive sentiment (1)
        data['review_score'] = data['review_score'].replace({1: 0, 2: 0, 4: 1, 5: 1})

    return data

def split(data):
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