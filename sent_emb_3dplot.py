import pandas as pd
import numpy as np
import json
import random
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

MODEL = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def get_vectors_labels(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data['vectors'], data['labels']

def pca(sentence_vectors, dims):
    pca = PCA(n_components=dims)
    vis_dims = pca.fit_transform(np.array(sentence_vectors))
    
    return vis_dims

def sample(vecs, size):
    return random.sample(vecs, size)

def sub_plot_3d(ax, sentiment, vectors, color):

    # Plot each sample category individually such that we can set label name.
    x=vectors[:, 0]
    y=vectors[:, 1]
    z=vectors[:, 2]
    ax.scatter(x, y, zs=z, zdir='z', c=color, label=sentiment)
    
def sub_plot_2d(ax, sentiment, vectors, color):

    # Plot each sample category individually such that we can set label name.
    x=vectors[:, 0]
    y=vectors[:, 1]
    ax.scatter(x, y, c=color, label=sentiment)
    
def plot_3d(vec_dataset, datasets, colors):

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(projection='3d')
    
    projections = []
    
    for vec in vec_dataset:
        projections.append(pca(vec, 3))
        
    for i, proj in enumerate(projections):
        sub_plot_3d(ax, datasets[i], proj, colors[i])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(bbox_to_anchor=(1.1, 1))
    
    plt.savefig('C:/4170_images/fig17.png')
    
def plot_2d(vec_dataset, datasets, colors):

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    
    projections = []
    
    for vec in vec_dataset:
        projections.append(pca(vec, 3))
        
    for i, proj in enumerate(projections):
        sub_plot_2d(ax, datasets[i], proj, colors[i])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(bbox_to_anchor=(1.1, 1))
    ax.legend(bbox_to_anchor=(1.1, 1))
    
    plt.savefig('C:/4170_images/fig16.png')
    
def plot_all(size, dims, datasets):
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    vec_ds = []

    for ds in datasets:
        X, _ = get_vectors_labels(f"C:/data/vectors/sentence/{ds}_test.json.gz")
        vec_ds.append(sample(X, size))

    if dims == 3:
        plot_3d(vec_ds, datasets, COLORS[:len(datasets)])
    else:
        plot_2d(vec_ds, datasets, COLORS[:len(datasets)])

DATASETS = ['steam', 'yelp', 'amazon', 'twitter', 'reddit']

plot_all(70, 3, DATASETS)