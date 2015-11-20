import csv
import time
import random
from itertools import chain

import logging
logger = logging.getLogger('Contextual')

from scipy import sparse
from scipy.sparse.linalg import svds
import numpy as np

def movielens_load():
    fpr = open('ratings.csv')
    fpt = open('tags.csv')
    gr = csv.reader(fpr, delimiter=',', quotechar='"')
    gt = csv.reader(fpt, delimiter=',', quotechar='"')
    gr.__next__()
    gt.__next__()
    return chain(gr, gt)

portion = 0.5
d = 30
gamma = 0.95
cox = []
coy = []
records = {}
for idx, (user, movie, rate, timestamp) in enumerate(movielens_load()):
    user = int(user) - 1
    movie = int(movie) - 1
    if random.uniform(0,1) < portion:
        cox.append(user)
        coy.append(movie)
    else:
        if user in records:
            records[user].append(movie)
        else:
            records[user] = [movie]

cox = np.array(cox)
coy = np.array(coy)
A = sparse.coo_matrix((np.ones(cox.shape), (cox, coy)), shape=(cox.max() + 1, coy.max() + 1), dtype=np.float32)
U, S, VT = svds(A, d)
for i in range(U.shape[0]):
    U[i] = U[i] / np.linalg.norm(U[i])

eligable_users = set(records.keys())
logger.info('#Total eligabli users = {0}'.format(len(eligable_users)))
logger.info('#Users participated ea/round = {0}'.format(1))
user = [random.sample(eligable_users, 1)[0]]
def environment(*arg):
    if len(arg) == 0:
        user[0] = random.sample(eligable_users, 1)[0]
        logger.debug('Random user {0}'.format(user))
        return U[user[0]]
    else:
        recommend = arg[0]
        movies = records[user[0]]
        logger.debug('Received recommendation {0}'.format(recommend))
        logger.debug('User ctr history {0}'.format(movies))
        for idx, movie in enumerate(recommend):
            if movie in movies:
                return idx, gamma ** idx
        return float('Inf'), 0
logger.info('Initializing environment "Movielens" done')
