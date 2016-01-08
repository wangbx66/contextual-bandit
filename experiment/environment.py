import heapq
import random
import logging
logger = logging.getLogger('Environment')

import numpy as np

from utils import red
from utils import uni
from utils import disturb
from utils import reward
from utils import ucb_settings

from isp import isp_oracle
from isp import reachable

def contextual_monkey_rng(L=20, d=10, h=0.35, K=4, gamma=0.95, eps=0.1, v=0.35, disj=False):
    arms = list(range(L))
    x = {arm: np.random.uniform(0, 1, d) for arm in arms}
    theta = h * uni(np.random.uniform(0, 1, d))
    logger.info('Initializing random settings "Contextual Monkey" complete')
    s = ucb_settings(arms=arms, L=L, x=x, theta=theta, K=K, d=d, gamma=gamma, eps=eps, v=v, disj=disj)
    logger.info(s)
    return s

def argmax_oracle(U, K, sort):
    return [p[1] for p in heapq.nlargest(K, [(U[arm], arm) for arm in U])[::2*sort-1]]

def contextual_monkey(s, cascade, rgamma, sort):
    logger.info('Initializing environment "Contextual Monkey"| cascade:{0} rgamma:{1} sort:{2}'.format(cascade, rgamma, sort))
    xt = {arm: s.x[arm] for arm in s.x}
    def environment(recommend=None):
        if recommend == None:
            for arm in s.x:
                xt[arm] = uni(s.x[arm] + np.random.uniform(-s.v, s.v, s.d))
            return xt, (s.K, sort)
        else:
            ctr = [np.random.uniform(0, 1) < s.theta.dot(xt[arm]) + np.random.normal(0, s.eps) for arm in recommend]
            r, c = reward(ctr, s.gamma, s.disj)
            return (r, c) if cascade else (r, [int(click) for click in ctr])
    logger.info('Initializing environment "Contextual Monkey" complete')
    return environment, ucb_settings(arms=s.arms, L=s.L, d=s.d, gamma=1-rgamma*(1-s.gamma), disj=s.disj, cascade=cascade, oracle=argmax_oracle, theta=s.theta)

def contextual_movielens(s, cascade, rgamma, sort):
    logger.info('Initializing environment "Contextual Movielens"| cascade:{0} rgamma:{1} sort:{2}'.format(cascade, rgamma, sort))
    user = random.sample(s.users, 1)
    def environment(recommend=None):
        if recommend is None:
            user[0] = random.sample(s.users, 1)[0]
            exc = s.A.getrow(user[0])
            if len([arm for arm in s.arms if exc[0, arm] == 0]) < s.K + 2:
                return environment()
            return {arm: np.outer(s.U[user[0]], s.V[arm]).flatten() for arm in s.arms if exc[0, arm] == 0}, (s.K, sort)
        else:
            ctr = [arm in s.ctrh[user[0]] for arm in recommend]
            r, c = reward(ctr, s.gamma, s.disj)
            return (r, c) if cascade else (r, [int(click) for click in ctr])
    logger.info('Initializing environment "Contextual Movielens" done')
    return environment, ucb_settings(arms=s.arms, L=s.L, d=s.d ** 2, gamma=1-rgamma*(1-s.gamma), disj=s.disj, cascade=cascade, oracle=argmax_oracle)

def contextual_isp(s, cascade, rgamma):
    logger.info('Initializing environment "Contextual ISP"| cascade:{0} rgamma:{1}'.format(cascade, rgamma))
    p = random.sample(s.G.nodes(), 2)
    xt = {arm: s.x[arm] for arm in s.x}
    def environment(recommend=None):
        if recommend is None:
            p[0], p[1] = random.sample(s.G.nodes(), 2)
            if not reachable(s.G, p[0], p[1]):
                return environment()
            for arm in s.x:
                xt[arm] = disturb(s.x[arm], s.h)
            return xt, (s.G, p[0], p[1])
        else:
            ctr = [s.tlc > np.random.exponential(1 - s.theta.dot(xt[arm])) for arm in recommend]
            r, c = reward(ctr, s.gamma, s.disj)
            return (r, c) if cascade else (r, [int(click) for click in ctr])
    logger.info('Initializing environment "Contextual ISP" done')
    return environment, ucb_settings(arms=s.G.edges(), L=len(s.G.edges()), d=s.d, gamma=1-rgamma*(1-s.gamma), disj=s.disj, cascade=cascade, oracle=isp_oracle, theta=s.theta)
