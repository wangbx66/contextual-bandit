import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import logging
logger = logging.getLogger('Environment')

seed = int(time.time() * 100) % 339
np.random.seed(seed)
logger.info('Numpy Random seed = {0}'.format(seed))

from functools import reduce

import colorama
colorama.init()

def red(s):
    return(colorama.Fore.RED + s.__str__() + colorama.Fore.WHITE)

def uni(x):
    return x/np.sqrt(x.dot(x))

def reward(ctr, gamma, disj):
    for c, click in enumerate(ctr):
        if click ^ disj:
            return int(disj) + gamma ** c * (int(not disj) - int(disj)), c
    return int(disj), len(ctr)

class ucb_settings:
    def __init__(self, **s):
        self.__dict__ = s
        
    def __str__(self):
        return '\n'.join([str(k) + ' ' + str(v) for k, v in self.__dict__.items()])

def contextual_monkey_rng(L=20, d=10, h=0.35, K=4, gamma=0.95, eps=0.1, v=0.35, disj=False):
    arms = list(range(L))
    x = {arm:np.random.uniform(0, 1, d) for arm in arms}
    theta = h * uni(np.random.uniform(0, 1, d))
    return ucb_settings(arms=arms, L=L, x=x, theta=theta, K=K, d=d, gamma=gamma, eps=eps, v=v, disj=disj)

def contextual_cascading_monkey(s=contextual_monkey_rng()):
    logger.info('Initializing environment "Contextual Cascading Monkey"')
    xt = {arm:s.x[arm] for arm in s.x}
    def environment(recommend=None):
        if recommend == None:
            for arm in xt:
                xt[arm] = uni(s.x[arm] + np.random.uniform(-s.v, s.v, s.d))
            return xt
        else:
            ctr = [np.random.uniform(0, 1) < s.theta.dot(xt[arm]) + np.random.normal(0, s.eps) for arm in recommend]
            r, c = reward(ctr, s.gamma, s.disj)
            return r, c
    logger.info('Initializing environment "Contextual Cascading Monkey" complete')
    return environment, ucb_settings(arms=s.arms, L=s.L, K=s.K, d=s.d, gamma=s.gamma, theta=s.theta, disj=s.disj)

def contextual_full_monkey(s=contextual_monkey_rng()):
    logger.info('Initializing environment "Contextual Full Monkey"')
    xt = {arm:s.x[arm] for arm in s.x}
    def environment(recommend=None):
        if recommend == None:
            for arm in xt:
                xt[arm] = uni(s.x[arm] + np.random.uniform(-s.v, s.v, s.d))
            return xt
        else:
            ctr = [(np.random.uniform(0, 1) < s.theta.dot(xt[arm]) + np.random.normal(0, s.eps)) for arm in recommend]
            r, _ = reward(ctr, s.gamma, s.disj)
            return r, [int(click) for click in ctr]
    logger.info('Initializing environment "Contextual Full Monkey" complete')
    return environment, ucb_settings(arms=s.arms, L=s.L, K=s.K, d=s.d, gamma=s.gamma, theta=s.theta)

def movielens(candidates, userno=1):
    logger.info('Initializing environment "Movielens"')
    records = {}
    with open('ratings.csv') as fp:
        r = csv.reader(fp, delimiter=',', quotechar='"')
        for idx, tpl in enumerate(r):
            if idx == 0:
                continue
            user, movie, rate, _ = tpl
            if not int(movie) in candidates:
                continue 
            try:
                records[int(user)].append(int(movie))
            except:
                records[int(user)] = [int(movie)]
    with open('tags.csv') as fp:
        r = csv.reader(fp, delimiter=',', quotechar='"')
        for idx, tpl in enumerate(r):
            if idx == 0:
                continue
            user, movie, tag, _ = tpl
            if not int(movie) in candidates:
                continue
            try:
                records[int(user)].append(int(movie))
            except:
                records[int(user)] = [(int(movie))]

    total = len(records)
    # leave those who have viewed at least one candidates
    eligable_users = set(records.keys())
    logger.info('#Total eligabli users = {0}'.format(len(eligable_users)))
    logger.info('#Users participated ea/round = {0}'.format(userno))
    def environment(recommend):
        users = random.sample(eligable_users, userno)
        movies = reduce(set.union, [set(records[user]) for user in users])
        logger.debug('Received recommendation {0}'.format(recommend))
        logger.debug('User ctr history {0}'.format(movies))
        for idx, movie in enumerate(recommend):
            if movie in movies:
                return idx
        return float('Inf')
    logger.info('Initializing environment "Movielens" done')
    return environment

def movielens_chrono():
    print('Initializing environment "Movielens-chrono"')
    records = {}
    tts = []
    with open('ratings.csv') as fp:
        r = csv.reader(fp, delimiter=',', quotechar='"')
        for idx, tpl in enumerate(r):
            if idx == 0:
                continue
            user, movie, rate, t = tpl
            tts.append(int(t))
            try:
                records[int(user)].append((int(t), int(movie)))
            except:
                records[int(user)] = [((int(t), int(movie)))]
    with open('tags.csv') as fp:
        r = csv.reader(fp, delimiter=',',quotechar='"')
        for idx, tpl in enumerate(r):
            if idx == 0:
                continue
            user, movie, tag, t = tpl
            tts.append(int(t))
            try:
                records[int(user)].append((int(t), int(movie)))
            except:
                records[int(user)] = [((int(t), int(movie)))]

    stts = sorted(tts)
    start = stts[10000]
    end = stts[-1000]
    interval = 86400
    del tts
    del stts

    rng = np.random.RandomState(100)
    tt = max(records)
    for i in range(1, tt + 1):
        records[i] = sorted(records[i])

    t = [start]
    def environment(recommend):
        t[0] += interval
        user = rng.randint(tt) + 1
        movies = [x[1] for x in records[user] if x[1] < t[0]]
        print('Environment: Time: {0}'.format(time.strftime("%Y/%m/%d", time.localtime(t[0]))))
        print('Environment: Received recommendation {0}'.format(recommend))
        print('Environment: User ctr history {0}'.format(movies))
        for idx, movie in enumerate(recommend):
            if movie in movies:
                return idx
        return float('Inf')
    print('Initializing environment "Movielens-chrono" done')
    return environment
