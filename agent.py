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

def contextual_cascading_monkey(T=100000):
    def agent(environment):
        e, s = environment
        score = 0
        for t in range(1, T):
            recc = random.sample(s.arms, s.K)
            r, _ = e(recc)
            score += r
            if t % 10000 == 0:
                print('Agent: Score={0}/{1}'.format(score, t))
    return agent
        
def contextual_cascading_sherry(environment, T=100000):
    e, s = environment()
    delta = 0.9
    lamb = 0.1
    theta = np.zeros(s.d)
    beta = 0
    V = lamb * np.eye(s.d)
    ldV = np.log(np.linalg.det(V))
    X = np.zeros((1, s.d))
    Y = np.zeros(1)
    score = [0]
    for i in range (1, T):
        x = e()
        U = {arm:min(theta.dot(x[arm]) + beta * x[arm].dot(np.linalg.inv(V)).dot(x[arm]), 1) for arm in s.arms}
        recc = [p[1] for p in heapq.nlargest(s.K, [(U[arm], arm) for arm in s.arms])]
        r, c = e(recc)
        V += sum([s.gamma ** (2*k) * np.outer(x[recc[k]], x[recc[k]]) for k in range(min(s.K, c+1))])
        X = np.concatenate([X] + [s.gamma ** k * x[recc[k]].reshape(1, s.d) for k in range(min(s.K, c+1))])
        Y = np.concatenate([Y] + [s.gamma ** k * (k == c) * np.ones(1) for k in range(min(s.K, c+1))])
        theta = np.linalg.inv(X.T.dot(X) + lamb * np.eye(s.d)).dot(X.T.dot(Y))
        beta = np.sqrt(np.log(np.linalg.det(V))- ldV - 2 * np.log(delta)) + np.sqrt(lamb)
        score.append(score[-1] + r)
    return score, cosine(s.theta, theta) 

from environment import contextual_cascading_monkey
environment = contextual_cascading_monkey
agent = contextual_cascading_sherry
exploit, explore = agent(environment)

