# sentiment_analysis
COMP 4710 Final Project

# Dependencies

1. pandas
2. nltk
3. xgboost
4. sklearn
5. matplotlib
6. numpy
7. wordcloud

Run to install each depedency:
```
pip install <dependency>
```

# Datasets

1. [Steam](https://www.kaggle.com/datasets/andrewmvd/steam-reviews)
2. [Yelp](https://www.yelp.com/dataset)
3. [Amazon](https://jmcauley.ucsd.edu/data/amazon/)
4. [Reddit](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset?select%3DReddit_Data.csv)
5. [Twitter](https://www.kaggle.com/datasets/kazanova/sentiment140)

# Models

1. XGBoost Classifier
2. Multinomial Naive Bayes
3. Support Vector Machines
4. Lexicon Score Aggregation

# Usage

To save and clean datasets locally ensure filenames are downloaded with `.csv.gz` extension:

1. Configure correct paths in `load_data.py`
2. Run: `python load_data.py <dataset name> <output path>

Example: 
```
python load_paths.py steam "C:/data/steam.csv.gz"
```

To run any model:
```
python model.py <dataset name> <path to dataset (generated above) <vector encoding> <model name>
```

Example: models include "mnb", "svm", "xgb"
```
python model.py steam "C:/data/steam.csv.gz" "tf_idf" "xgb"
```

Analysis and visualizations can be produced with the following commands:
```
python analyze_dataset.py <dataset_name> 
python compare_datasets.py <dataset_name> <vector_encoding>
python sent_emb_3dplot.py
python word_embeddings.py
```

There is also an included Excel file with some additional charts plotting accuracies and cosine similarities as seen in the paper found in `excel/analysis.xlsx`
