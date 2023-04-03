import pandas as pd
import sys
from sklearn.model_selection import train_test_split

STEAM_PATH = 'C:/data/dataset.csv.gz'
YAHOO_PATH = 'C:/users/emers/downloads/yelp_academic_dataset_review.json'


def batch_load(path, is_csv, text_column, score_column):
    # load file in batches, take text and scores

    data = []

    if is_csv:
        for chunk in pd.read_csv(path, compression='gzip', chunksize=100000):

            # Process each chunk of data here
            data.append(chunk[[text_column, score_column]])
    else:
        for chunk in pd.read_json(path, lines=True, chunksize=100000):
            # Process each chunk of data here

            data.append(chunk[[text_column, score_column]])

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

def load_steam():

    data = batch_load(STEAM_PATH, True, 'review_text', 'review_score')
    data = sample(data)

    # replace -1 with 0
    data.loc[:, "review_score"] = (
        data["review_score"]
         .apply(lambda x: 0 if x == -1 else x)
    )

    return data

def load_yelp():

    data = batch_load(YAHOO_PATH, False, 'text', 'stars')
    data = sample(data)

    # remove 3 start reviews (neutral sentiment)
    data = data[data['review_score'] != 3]

    # map 1 and 2 start reviews to a negative sentiment (0)
    # map 4 and 5 start reviews to a positive sentiment (1)
    data['review_score'] = data['review_score'].replace({1: 0, 2: 0, 4: 1, 5: 1})

    return data

def save_train_test(dataset, out_path):

    load_functions = {
        'steam': load_steam,
        'yelp': load_yelp
    }

    if dataset in load_functions.keys():

        print("Loading dataset...")
        data = load_functions[dataset]()

        print("Writing dataset to disk...")
        data.to_csv(out_path, compression='gzip', index=False)

        print(f"File saved to {out_path}")
    else:
        print("Dataset does not exist")


if __name__ == '__main__':
    if len(sys.argv) == 3:

        save_train_test(sys.argv[1], sys.argv[2])
    else:
        print("Please provide two arguments")