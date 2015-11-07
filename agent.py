import math
import csv
import heapq

import numpy as np
import matplotlib.pyplot as plt

import time
import random
seed = time.time()
random.seed(seed)
import logging
logging.basicConfig(format='%(name)s-%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger('Agent')
logger.info('Random seed = {0}'.format(seed))

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
                logger.info('Reward={0}'.format(reward))
                logger.info('Score={0}/{1}'.format(score, t))
            for k in range(min(c, K)):
                ob[recc[k]] += 1
                w[recc[k]] = ((ob[recc[k]] - 1) * w[recc[k]] + (c == k)) / ob[recc[k]]
        logger.info('Reward={0}'.format(reward))
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
                print('Agent: Reward={0}'.format(reward))
                print('Agent: Score={0}/{1}'.format(score, t))
            else:
                if t % 10000 == 0:
                    print('Agent: Reward={0}'.format(reward))
                    print('Agent: Score={0}/{1}'.format(score, t))
    return agent

from environment import movielens
candidates = movies_mix_cld_adt()
environment = movielens(candidates[0] + candidates[1], 5)
agent = ucb(candidates)
agent(environment)
agent = monkey(candidates)
agent(environment)

