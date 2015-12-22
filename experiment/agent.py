import time
import heapq
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

import random
import logging
logger = logging.getLogger('Agent')

def contextual_cascading_monkey(e, s, T):
    score = [0]
    for t in range(1, T):
        _ = e()
        recc = random.sample(s.arms, s.K)
        r, _ = e(recc)
        score.append(score[-1] + r)
    logger.info('Monkey play score {0}/{1}'.format(score[-1], T))
    return score, 0

contextual_full_monkey = contextual_cascading_monkey

def absolute_cascading_ucb(e, s, T):
    score = [0]
    W = {arm:0.5 for arm in s.arms}
    b = {arm:1 for arm in s.arms}
    for t in range(1, T):
        _ = e()
        U = {arm:W[arm] + np.sqrt(1.5 * np.log(t) / b[arm]) for arm in s.arms}
        recc = [p[1] for p in heapq.nlargest(s.K, [(U[arm], arm) for arm in s.arms])]
        r, c = e(recc)
        for k in range(min(s.K, c+1)):
            W[recc[k]] = ((b[recc[k]]) * W[recc[k]] + ((k == c) == s.disj)) / (b[recc[k]] + 1)
            b[recc[k]] += 1
        score.append(score[-1] + r)
    logger.info('Absolute play score {0}/{1}'.format(score[-1], T))
    return score, 0

def contextual_cascading_sherry(e, s, T):
    delta = 0.9
    lamb = 0.1
    theta = np.zeros(s.d)
    beta = 0
    V = lamb * np.eye(s.d)
    ldV = np.linalg.slogdet(V)[1]
    X = np.zeros((1, s.d))
    Y = np.zeros(1)
    score = [0]
    timestamp = time.time()
    for t in range (1, T):
        x = e()
        U = {arm:theta.dot(x[arm]) + beta * x[arm].dot(np.linalg.inv(V)).dot(x[arm]) for arm in x}
        recc = [p[1] for p in heapq.nlargest(s.K, [(U[arm], arm) for arm in x])]
        r, c = e(recc)
        V += sum([s.gamma ** (2*k) * np.outer(x[recc[k]], x[recc[k]]) for k in range(min(s.K, c+1))])
        X = np.concatenate([X] + [s.gamma ** k * x[recc[k]].reshape(1, s.d) for k in range(min(s.K, c+1))])
        Y = np.concatenate([Y] + [s.gamma ** k * ((k == c) == s.disj) * np.ones(1) for k in range(min(s.K, c+1))])
        theta = np.linalg.inv(X.T.dot(X) + lamb * np.eye(s.d)).dot(X.T.dot(Y))
        beta = np.sqrt(np.linalg.slogdet(V)[1] - ldV - 2 * np.log(delta)) + np.sqrt(lamb)
        score.append(score[-1] + r)
        if t % 500 == 0:
            logger.debug('Sherry {0} rounds with {1}s elapsed'.format(t, int(time.time() - timestamp)))
    logger.info('Sherry play score {0}/{1}'.format(score[-1], T))
    if 'theta' in s.__dict__:
        logger.info('theta cosine similarity {0}'.format(1 - cosine(s.theta, theta)))
        return score, 1 - cosine(s.theta, theta)
    else:
        return score, 0

def contextual_full_lijing(e, s, T):
    delta = 0.9
    lamb = 0.1
    theta = np.zeros(s.d)
    beta = 0
    V = lamb * np.eye(s.d)
    b = np.zeros(s.d)
    score = [0]
    for t in range (1, T):
        x = e()
        U = {arm:theta.dot(x[arm]) + beta * x[arm].dot(np.linalg.inv(V)).dot(x[arm]) for arm in x}
        recc = [p[1] for p in heapq.nlargest(s.K, [(U[arm], arm) for arm in x])]
        r, ctr = e(recc)
        V += sum([s.gamma ** (2*k) * np.outer(x[recc[k]], x[recc[k]]) for k in range(len(recc))])
        b += sum([ctr[k] * x[recc[k]] for k in range(len(recc))])
        theta = np.linalg.inv(V).dot(b)
        beta = np.sqrt(s.d * np.log((1 + t*s.L) / delta)) + np.sqrt(lamb)
        score.append(score[-1] + r)
    logger.info('Lijing play score {0}/{1}'.format(score[-1], T))
    if 'theta' in s.__dict__:
        logger.info('theta cosine similarity {0}'.format(1 - cosine(s.theta, theta)))
        return score, 1 - cosine(s.theta, theta)
    else:
        return score, 0
