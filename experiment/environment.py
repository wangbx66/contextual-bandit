import numpy as np
import random
import logging
logger = logging.getLogger('Environment')

from functools import reduce

from utils import red
from utils import uni
from utils import reward
from utils import ucb_settings
from movielens import contextual_movielens_rng

def contextual_monkey_rng(L=20, d=10, h=0.35, K=4, gamma=0.95, eps=0.1, v=0.35, disj=False):
    arms = list(range(L))
    x = {arm:np.random.uniform(0, 1, d) for arm in arms}
    theta = h * uni(np.random.uniform(0, 1, d))
    logger.info('Initializing random settings "Contextual Monkey" complete')
    s = ucb_settings(arms=arms, L=L, x=x, theta=theta, K=K, d=d, gamma=gamma, eps=eps, v=v, disj=disj)
    logger.info(s)
    return s

def contextual_cascading_monkey(s):
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

def contextual_full_monkey(s):
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

def contextual_cascading_movielens(s):
    logger.info('Initializing environment "Contextual Cascading Movielens"')
    user = random.sample(s.users, 1)
    def environment(recommend=None):
        if recommend is None:
            user[0] = random.sample(s.users, 1)[0]
            exc = s.A.getrow(user[0])
            if len([arm for arm in s.arms if exc[0, arm] == 0]) < s.K + 2:
                return environment()
            return {arm: np.outer(s.U[user[0]], s.V[arm]).flatten() for arm in s.arms if exc[0, arm] == 0}
        else:
            ctr = [arm in s.ctrh[user[0]] for arm in recommend]
            r, c = reward(ctr, s.gamma, s.disj)
            return r, c
    logger.info('Initializing environment "Contextual Cascading Movielens" done')
    return environment, ucb_settings(arms=s.arms, L=s.L, K=s.K, d=s.d ** 2, gamma=s.gamma, disj=s.disj)

def contextual_full_movielens(s):
    logger.info('Initializing environment "Contextual Full Movielens"')
    user = random.sample(s.users, 1)
    def environment(recommend=None):
        if recommend is None:
            user[0] = random.sample(s.users, 1)[0]
            exc = s.A.getrow(user[0])
            if len([arm for arm in s.arms if exc[0, arm] == 0]) < s.K + 2:
                return environment()
            return {arm: np.outer(s.U[user[0]], s.V[arm]).flatten() for arm in s.arms if exc[0, arm] == 0}
        else:
            ctr = [arm in s.ctrh[user[0]] for arm in recommend]
            r, _ = reward(ctr, s.gamma, s.disj)
            return r, [int(click) for click in ctr]
    logger.info('Initializing environment "Contextual Full Movielens" done')
    return environment, ucb_settings(arms=s.arms, L=s.L, K=s.K, d=s.d ** 2, gamma=s.gamma, disj=s.disj)
