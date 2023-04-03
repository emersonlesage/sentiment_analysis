import pandas as pd
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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

    # drop early access reviews
    df = df[df['review_text'] != 'early access review']
    
    # drop empty reviews
    df = df[(df['review_text'] != ' ') & (df['review_text'] != '')]

    return df


def tf_idf(train, test):
    vectorizer = TfidfVectorizer()

    vectorizer.fit(pd.concat([train, test]))

    return vectorizer.transform(train), vectorizer.transform(test)

def preprocess_train_test_sets(train_path, test_path, vector_encoding):

    vector_encoding_functions = {
        'tf_idf': tf_idf
    }

    if vector_encoding in vector_encoding_functions.keys():
        train = pd.read_csv(train_path, compression='gzip')
        test = pd.read_csv(test_path, compression='gzip')

        X_train, X_test = vector_encoding_functions[vector_encoding](train['review_text'], test['review_text'])
        
        y_train = train['review_score']
        y_test = test['review_score']

        return X_train, X_test, y_train, y_test

    else:
        print("Vector encoding does not exist.")

        return None