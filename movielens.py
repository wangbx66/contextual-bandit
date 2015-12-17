import csv
import random
from itertools import chain

import logging
logger = logging.getLogger('Movielens')

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from utils import ucb_settings
from utils import overlap

def movielens_data():
    fpr = open('ratings.csv')
    fpt = open('tags.csv')
    gr = csv.reader(fpr, delimiter=',', quotechar='"')
    gt = csv.reader(fpt, delimiter=',', quotechar='"')
    gr.__next__()
    gt.__next__()
    return chain(gr, gt)

def contextual_movielens_rng(L=None, portion=0.5, d=15, K=6, h=None, gamma=0.95, disj=False):
    cox = []
    coy = []
    history = {}
    movies = {}
    for idx, (user, movie, rate, timestamp) in enumerate(movielens_data()):
        user = int(user) - 1
        movie = int(movie) - 1
        if movie in movies:
            movies[movie] += 1
        else:
            movies[movie] = 1
        if np.random.uniform(0,1) > portion:
            cox.append(user)
            coy.append(movie)
        else:
            if user in history:
                history[user].append(movie)
            else:
                history[user] = [movie]
    A = sparse.coo_matrix((np.ones(len(cox)), (cox, coy)), shape=(max(list(history) + cox) + 1, max(list(movies) + coy) + 1), dtype=np.float32)
    U, S, VT = svds(A, d)
    for i in range(U.shape[0]):
        U[i] = U[i] / np.linalg.norm(U[i])
    V = VT.T
    for i in range(V.shape[0]):
        V[i] = V[i] / np.linalg.norm(V[i])

    if L is None:
        L = len(movies)
    if h is None:
        h = len(history)
    selected_movies = set([x[1] for x in sorted([(movies[movie], movie) for movie in movies])[-L:]])
    eligable_users = [x[1] for x in sorted([(overlap(history[user], selected_movies), user) for user in history])[-h:]]
    logger.info('Initializing random settings "Contextual Movielens" complete')
    s = ucb_settings(L=L, d=d, K=K, gamma=gamma, disj=disj, users=eligable_users, arms=selected_movies, ctrh=history, U=U, V=V)
    logger.info(s)
    return s
