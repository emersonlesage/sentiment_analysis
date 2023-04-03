# sentiment_analysis
COMP 4710 Final Project

# Dependencies

1. pandas
2. nltk
3. xgboost
4. sklearn

# Datasets

1. Steam (COMPLETE)
2. Yelp (COMPLETE)
3. Amazon (IN PROGRESS)

# Models

1. XGBoost Classifier (IN PROGRESS)
2. Random Forest (NOT STARTED)

# Usage

Avaliable datasets: steam, yelp

Avaliable vector encodings: tf_idf

Install dependencies with pip

To save and clean datasets locally:

Filename must have .csv.gz extension

1. configure paths in load_paths.py
2. python load_data.py <dataset name> <output path>

example: python load_paths.py steam "C:/data/steam.csv.gz"

To run any model:
```
python model.py <dataset name> <path to dataset (generated above) <vector encoding> <model name>
```

Example: models include "mnb", "svm", "xgb"
```
python model.py steam "C:/data/steam.csv.gz" "tf_idf" "xgb"
```
