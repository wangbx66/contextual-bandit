import csv
import time
import random
from itertools import chain

import logging

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

def movielens_data():
    fpr = open('ratings.csv')
    fpt = open('tags.csv')
    gr = csv.reader(fpr, delimiter=',', quotechar='"')
    gt = csv.reader(fpt, delimiter=',', quotechar='"')
    gr.__next__()
    gt.__next__()
    return chain(gr, gt)

def contextual_movielens_rng(L=None, portion=0.5, d=15, K=6, gamma=0.95, disj=False):
    cox = []
    coy = []
    history = {}
    movies = {}
    for idx, (user, movie, rate, timestamp) in enumerate(movielens_load()):
        user = int(user) - 1
        movie = int(movie) - 1
        if movie in movies
            movies[movie] += 1
        else:
            movies[movie] = 1
        if random.uniform(0,1) < portion:
            cox.append(user)
            coy.append(movie)
        else:
            if user in history:
                history[user].append(movie)
            else:
                history[user] = [movie]
    cox = np.array(cox)
    coy = np.array(coy)
    A = sparse.coo_matrix((np.ones(cox.shape), (cox, coy)), shape=(cox.max() + 1, coy.max() + 1), dtype=np.float32)
    U, S, VT = svds(A, d)
    for i in range(U.shape[0]):
        U[i] = U[i] / np.linalg.norm(U[i])
    V = VT.T
    for i in range(V.shape[0]):
        V[i] = V[i] / np.linalg.norm(V[i])
    
    if L is None:
        L = len(movies)
    selected_movies = [x[1] for x in sorted[(movies[movie], movie) for movie in movies][-L:]]
    eligable_users = set(history.keys())
print('#Total eligabli users = {0}'.format(len(eligable_users)))
print('#Users participated ea/round = {0}'.format(1))
user = random.sample(eligable_users, 1)
def environment(*arg):
    if len(arg) == 0:
        user[0] = random.sample(eligable_users, 1)[0]
        logger.debug('Random user {0}'.format(user))
        return U[user[0]]
    else:
        recommend = arg[0]
        movies = history[user[0]]
        logger.debug('Received recommendation {0}'.format(recommend))
        logger.debug('User ctr history {0}'.format(movies))
        for idx, movie in enumerate(recommend):
            if movie in movies:
                return idx, gamma ** idx
        return float('Inf'), 0
logger.info('Initializing environment "Movielens" done')
