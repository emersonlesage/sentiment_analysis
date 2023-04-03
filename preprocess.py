import pandas as pd
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from load_data import load_steam
from load_data import load_yelp

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

STOP_WORDS = nltk.corpus.stopwords.words('english')
LEMMATIZER = nltk.stem.WordNetLemmatizer()

def remove_stopwords(text):
    # remove words such as "and", "is", "an", etc.

    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in STOP_WORDS]
    return ' '.join(filtered_tokens)

def lemmatize(text):
    # reduce words to base component. "Learning" becomes "Learn", "Going" becomes "Go"

    # Split the text into individual words
    words = text.split()
    
    # Lemmatize each word in the list of words
    lemmatized_words = [LEMMATIZER.lemmatize(word) for word in words]
    
    # Join the lemmatized words back into a string
    lemmatized_text = ' '.join(lemmatized_words)
    
    return lemmatized_text

def lowercase(text):
    # converts text to lowercase
    
    return text.lower()

def remove_non_ascii(text):
    # removes non ASCII characters from a string
    
    alphabet_pattern = r'[^a-z]+'
    return re.sub(alphabet_pattern, ' ', text)

def preprocess(df):
    # apply all preprocessing to dataframe

    # clean 
    df.loc[:, "review_text"] = (
        df["review_text"]
        .apply(lowercase)
        .apply(remove_non_ascii)
        .apply(remove_stopwords)
        .apply(lemmatize)
    )

    return df

def preprocess_steam(path):

    df = pd.read_csv(path, compression='gzip')
    df = preprocess(df)

    # drop early access reviews
    df = df[df['review_text'] != 'early access review']
    
    # drop empty reviews
    df = df[(df['review_text'] != ' ') & (df['review_text'] != '')]
    
    return df

def preprocess_yelp(path):

    df = pd.read_csv(path, compression='gzip')
    df = preprocess(df)

    return df

def tf_idf(text):
    vectorizer = TfidfVectorizer()

    return vectorizer.fit_transform(text)

def get_train_test_sets(dataset, data_path, vector_encoding):
    
    preprocessing_functions = {
        'steam': preprocess_steam,
        'yelp': preprocess_yelp
    }

    vector_encoding_functions = {
        'tf_idf': tf_idf
    }

    if dataset in preprocessing_functions.keys() and vector_encoding in vector_encoding_functions.keys():

        data = preprocessing_functions[dataset](data_path)

        X = vector_encoding_functions[vector_encoding](data['review_text'])
        y = data['review_score']

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

        return X_train, X_test, y_train, y_test

    else:
        print("Dataset or vector encoding does not exist.")

        return None