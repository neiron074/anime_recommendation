import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import sigmoid_kernel

def rating_recommend(data, anime_name, n_neighbors=11, flag = False):


    pivot = pd.pivot_table(data, values='rating', index='Name', columns='user_id')
    pivot = pivot.fillna(0)

    if flag == True:
        n_neighbors = pivot.shape[0]

    anime_matrix = csr_matrix(pivot.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(anime_matrix)

    for i, name in enumerate(pivot.index):
        if name == anime_name:
            index = i

    distances, indices = model_knn.kneighbors(pivot.iloc[index, :].values.reshape(1, -1), n_neighbors=n_neighbors)

    result = pd.DataFrame({'Name': pivot.index[indices.flatten()[:n_neighbors]], 'dist':distances.flatten()[:n_neighbors]})
    return result


def genre_recommend(data, name, n_neighbors=11, flag = False):

    if flag == True:
        n_neighbors = data.shape[0]

    tfid = TfidfVectorizer(min_df=3, max_features=None,
                           strip_accents='unicode', analyzer='word',
                           ngram_range=(1, 1),
                           stop_words='english')
    tfid_data = tfid.fit_transform(data['Genres'].values)

    sig = sigmoid_kernel(tfid_data, tfid_data)

    indices = pd.Series(data.index, index=data['Name']).drop_duplicates()

    idx = indices[name]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    anime_indices = []
    scores = []
    for i, index in enumerate(sig_scores):
        if i == n_neighbors:
            break
        scores.append(1 - index[1])
        anime_indices.append(index[0])
    result = pd.DataFrame({'Name': data['Name'].iloc[anime_indices].values, 'dist': scores})
    return result


def sypnopsis_recommend(data, name, n_neighbors=11, flag = False):

    if flag == True:
        n_neighbors = data.shape[0]

    tfid = TfidfVectorizer(min_df=3, max_features=None,
                           strip_accents='unicode', analyzer='word',
                           ngram_range=(1, 1),
                           stop_words='english')
    tfid_data = tfid.fit_transform(data['sypnopsis'].values)

    sig = sigmoid_kernel(tfid_data, tfid_data)

    indices = pd.Series(data.index, index=data['Name']).drop_duplicates()

    idx = indices[name]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    anime_indices = []
    scores = []
    for i, index in enumerate(sig_scores):
        if i == n_neighbors:
            break
        scores.append(1 - index[1])
        anime_indices.append(index[0])
    result = pd.DataFrame({'Name': data['Name'].iloc[anime_indices].values, 'dist': scores})
    return result


def recommend(anime, rating, name, n_neighbors=11):
    rating_rec = rating_recommend(rating, name, flag=True)
    genre_rec = genre_recommend(anime, name, flag=True)
    syn_rec = sypnopsis_recommend(anime, name, flag=True)

    result = pd.DataFrame()

    for i, name in enumerate(rating_rec['Name']):
        result.loc[i, 'Name'] = name
        result.loc[i, 'dist'] = (rating_rec.loc[rating_rec['Name'] == name, 'dist'].values[0] +
                                 genre_rec.loc[genre_rec['Name'] == name, 'dist'].values[0] +
                                 syn_rec.loc[syn_rec['Name'] == name, 'dist'].values[0]) / 3

    return result.sort_values(by='dist')[:n_neighbors]