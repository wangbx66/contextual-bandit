import math
import csv
import heapq

import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

import time
import random

import logging
logging.basicConfig(format='%(name)s-%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger('Agent')

from environment import contextual_cascading_monkey as contextual_cascading_monkey_environment
from environment import contextual_full_monkey as contextual_full_monkey_environment
from environment import contextual_monkey_rng

def movies_mix_cld_adt(L=200):
    children, adults = set(), set()
    with open('movies.csv') as fp:
        r = csv.reader(fp, delimiter=',',quotechar='"')
        for i, tpl in enumerate(r):
            if i == 0:
                continue
            idx, movie, genres = tpl
            genres = genres.split('|')
            if 'Children' in genres:
                children.add(int(idx))
            else:
                adults.add(int(idx))
    children = random.sample(children, int(L/2))
    adults = random.sample(adults, int(L/2))
    movies = (children, adults)
    return movies

def ucb(candidates, K=8, T=150000, verbose=False):
    def agent(environment):
        score = 0
        children, adults = candidates
        movies = children + adults
        ob = {x:1 for x in movies}
        w = {x:0.5 for x in movies}
        u = {x:0 for x in movies}
        for t in range(1, T):
            for x in movies:
                u[x] = w[x] + math.sqrt(1.5 * math.log(t) / ob[x])
            recccld = heapq.nlargest(int(K/2), [(u[x],x) for x in children])
            reccadt = heapq.nlargest(int(K/2), [(u[x],x) for x in adults])
            recc = [p[1] for p in sorted(recccld + reccadt)]
            c = environment(recc)
            reward = int(isinstance(c, int))
            score += reward
            logger.debug('Reward={0}'.format(reward))
            logger.debug('Score={0}/{1}'.format(score, t))
            if t % 10000 == 0:
                logger.info('Score={0}/{1}'.format(score, t))
            for k in range(min(c, K)):
                ob[recc[k]] += 1
                w[recc[k]] = ((ob[recc[k]] - 1) * w[recc[k]] + (c == k)) / ob[recc[k]]
        logger.info('Score={0}/{1}'.format(score, t))
        print(sorted(u.items()))
        print(sorted(ob.items()))
        print(sorted(w.items()))
    return agent

def monkey(candidates, K=8, T=150000, verbose=False):
    def agent(environment):
        score = 0
        children, adults = candidates
        movies = children + adults
        for t in range(1, T):
            recc = random.sample(movies, K)
            c = environment(recc)
            reward = int(isinstance(c, int))
            score += reward
            if verbose:
                print('Agent: Score={0}/{1}'.format(score, t))
            else:
                if t % 10000 == 0:
                    print('Agent: Score={0}/{1}'.format(score, t))
    return agent

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
            W[recc[k]] = ((b[recc[k]]) * W[recc[k]] + ((k == c) ^ s.disj)) / (b[recc[k]] + 1)
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
    ldV = np.log(np.linalg.det(V))
    X = np.zeros((1, s.d))
    Y = np.zeros(1)
    score = [0]
    for t in range (1, T):
        x = e()
        U = {arm:theta.dot(x[arm]) + beta * x[arm].dot(np.linalg.inv(V)).dot(x[arm]) for arm in s.arms}
        recc = [p[1] for p in heapq.nlargest(s.K, [(U[arm], arm) for arm in s.arms])]
        r, c = e(recc)
        V += sum([s.gamma ** (2*k) * np.outer(x[recc[k]], x[recc[k]]) for k in range(min(s.K, c+1))])
        X = np.concatenate([X] + [s.gamma ** k * x[recc[k]].reshape(1, s.d) for k in range(min(s.K, c+1))])
        Y = np.concatenate([Y] + [s.gamma ** k * ((k == c) ^ s.disj) * np.ones(1) for k in range(min(s.K, c+1))])
        theta = np.linalg.inv(X.T.dot(X) + lamb * np.eye(s.d)).dot(X.T.dot(Y))
        beta = np.sqrt(np.log(np.linalg.det(V))- ldV - 2 * np.log(delta)) + np.sqrt(lamb)
        score.append(score[-1] + r)
    logger.info('Sherry play score {0}/{1}'.format(score[-1], T))
    logger.info('theta cosine similarity {0}'.format(1 - cosine(s.theta, theta)))
    return score, 1 - cosine(s.theta, theta)

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
        U = {arm:theta.dot(x[arm]) + beta * x[arm].dot(np.linalg.inv(V)).dot(x[arm]) for arm in s.arms}
        recc = [p[1] for p in heapq.nlargest(s.K, [(U[arm], arm) for arm in s.arms])]
        r, ctr = e(recc)
        V += sum([s.gamma ** (2*k) * np.outer(x[recc[k]], x[recc[k]]) for k in range(len(recc))])
        b += sum([ctr[k] * x[recc[k]] for k in range(len(recc))])
        theta = np.linalg.inv(V).dot(b)
        beta = np.sqrt(s.d * np.log((1 + t*s.L) / delta)) + np.sqrt(lamb)
        score.append(score[-1] + r)
    logger.info('Lijing play score {0}/{1}'.format(score[-1], T))
    logger.info('theta cosine similarity {0}'.format(1 - cosine(s.theta, theta)))
    return score, 1 - cosine(s.theta, theta)

def flowtest():
    T = 10000
    s = contextual_monkey_rng(L=20, d=10, h=0.75, K=4, gamma=0.95, eps=0.1, v=0.35, disj=True)
    exploit1, explore1 = contextual_cascading_sherry(*contextual_cascading_monkey_environment(s), T=T)
    exploit2, explore2 = contextual_cascading_monkey(*contextual_cascading_monkey_environment(s), T=T)
    exploit3, explore3 = contextual_full_monkey(*contextual_full_monkey_environment(s), T=T)
    exploit4, explore4 = contextual_full_lijing(*contextual_full_monkey_environment(s), T=T)
    exploit5, explore5 = absolute_cascading_ucb(*contextual_cascading_monkey_environment(s), T=T)
    plt.plot(range(T), exploit1, 'r--', range(T), exploit2, 'r--', range(T), exploit4, 'b--')
    #plt.show()

T = 10000
s = contextual_monkey_rng()
flowtest()

