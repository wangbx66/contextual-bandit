from itertools import chain
from collections import Counter
import csv
import random
import logging
logger = logging.getLogger('Movielens')

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from utils import uni
from utils import argmax_oracle
from utils import serialize

def overlap(l1, l2):
    return len(set(l1) & l2)

def movielens_data():
    fpr = open('movielens/ratings.csv', encoding='utf-8')
    fpt = open('movielens/tags.csv', encoding='utf-8')
    gr = csv.reader(fpr, delimiter=',', quotechar='"')
    gt = csv.reader(fpt, delimiter=',', quotechar='"')
    gr.__next__()
    gt.__next__()
    return chain(gr, gt)

class c3_movielens_rng:

    # c3synthetic_movielens_rng(n_movies=None, train_portion=0.5, d=15, K=6, n_users=None, gamma=0.95, disj=False)

    def __init__(self, **kwarg):
        logger.info('Initializing random settings "Contextual Movielens"')
        self.__dict__.update(kwarg)
        self.name = 'c3-movielens'
        self.oracle = argmax_oracle
        self.theta = None
        self.regret_avl = False
        self.load()
        logger.info(self)

    def __str__(self):
        return serialize(self, 'arms', 'x', 'ctrh', 'A', 'U', 'S',  'V', 'VT', 'users')

    def slot(self):
        self.user = random.sample(self.users, 1)[0]
        exc = self.A.getrow(self.user)
        #print(len([arm for arm in self.arms if exc[0, arm] == 0]))
        current = [arm for arm in self.arms if exc[0, arm] == 0]
        print(len(current), sum([arm in self.ctrh[self.user] for arm in current]))
        return {arm: np.outer(self.U[self.user], self.V[arm]).flatten() for arm in self.arms if exc[0, arm] == 0} if len([arm for arm in self.arms if exc[0, arm] == 0]) > self.K + 2 else self.slot()

    def realize(self, action):
        return [arm in self.ctrh[self.user] for arm in action]
    
    def regret(self, action):
        return 0
    
    def params(self, descend):
        return (self.K, descend)
    
    def load(self):
        cox = []
        coy = []
        self.ctrh = {}
        movies = {}
        for user, movie, rate, timestamp in movielens_data():
            user = int(user) - 1
            movie = int(movie) - 1
            if movie in movies:
                movies[movie] += 1
            else:
                movies[movie] = 1
            if np.random.uniform(0,1) < self.train_portion:
                cox.append(user)
                coy.append(movie)
            else:
                if user in self.ctrh:
                    self.ctrh[user].append(movie)
                else:
                    self.ctrh[user] = [movie]
        self.A = sparse.coo_matrix((np.ones(len(cox)), (cox, coy)), shape=(max(list(self.ctrh) + cox) + 1, max(list(movies) + coy) + 1), dtype=np.float32)
        self.U, self.S, self.VT = svds(self.A, self.d)
        self.A = self.A.astype(np.int32)
        for i in range(self.U.shape[0]):
            self.U[i] = uni(self.U[i])
        self.V = self.VT.T
        for i in range(self.V.shape[0]):
            self.V[i] = uni(self.V[i])
        if self.n_movies is None:
            self.L = len(movies)
        else:
            self.L = self.n_movies
        if self.n_users is None:
            self.n_users = len(self.ctrh)
        self.arms = set([x[1] for x in sorted([(movies[movie], movie) for movie in movies])[-self.L:]])
        self.users = [x[1] for x in sorted([(overlap(self.ctrh[user], self.arms), user) for user in self.ctrh if self.L*0.04 < overlap(self.ctrh[user], self.arms) < self.L*0.06])]#[-self.n_users:]]
        logging.info('total {0} users involved'.format(len(self.users)))
        self.d = self.d ** 2

    def ctrh_hist(self):
        hist = {user: len([movie for movie in self.ctrh[user] if movie in self.arms]) for user in self.ctrh}
        return Counter([v for k, v in hist.items()])

class c3_Lmovielens_rng(c3_movielens_rng):
    def __init__(self, **kwarg):
        logger.info('Initializing random settings "Contextual L-Movielens"')
        self.__dict__.update(kwarg)
        self.name = 'c3-L-movielens'
        self.oracle = argmax_oracle
        self.theta = None
        self.regret_avl = False
        self.load()
        self.d = kwarg['d']
        logger.info(self)
    
    def slot(self):
        self.user = random.sample(self.users, 1)[0]
        exc = self.A.getrow(self.user)
        return {arm: self.V[arm].flatten() for arm in self.arms if exc[0, arm] == 0} if len([arm for arm in self.arms if exc[0, arm] == 0]) < self.K + 2 else self.slot()
