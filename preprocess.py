import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

STOP_WORDS = nltk.corpus.stopwords.words('english')
LEMMATIZER = nltk.stem.WordNetLemmatizer()
MODEL = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


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

    print("\nPreprocessing dataset...", end='')

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

    print(f" DONE: {len(df)} records")

    return df


def tf_idf(train, test):
    vectorizer = TfidfVectorizer(max_features=1000)

    vectorizer.fit(train)

    return vectorizer.transform(train).toarray(), vectorizer.transform(test).toarray()

def sentence_embedding(train, test):

    train = MODEL.encode(train, batch_size=1500)
    test = MODEL.encode(test, batch_size=1500)

    return train, test

def vectorize(train, test, vector_encoding):

    vector_encoding_functions = {
        'tf_idf': tf_idf,
        'sentence': sentence_embedding
    }

    if vector_encoding in vector_encoding_functions.keys():

        print(f"\nVectorizing with {vector_encoding}")
        train_vec, test_vec = vector_encoding_functions[vector_encoding](train['review_text'], test['review_text'])

        train_json = {'vectors': train_vec.tolist(), 'labels': train['review_score'].tolist()}
        test_json = {'vectors': test_vec.tolist(), 'labels': test['review_score'].tolist()}

        return train_json, test_json

    else:
        print("Vector encoding does not exist.")

        return None