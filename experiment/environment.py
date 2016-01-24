
import random
import logging
logger = logging.getLogger('Environment')

import numpy as np

from utils import red
from utils import pendulum
from utils import suni
from utils import disturb
from utils import reward
from utils import ereward
from utils import ucb_settings
from utils import serialize
from utils import argmax_oracle

class c3synthetic_monkey_rng:

    # contextual_monkey_rng(self, L=20, d=10, b=0, K=4, gamma=0.95, eps=0.1, v=0.35, disj=False)
    
    def __init__(self, **kwarg):
        logger.info('Initializing random settings "Contextual Monkey"')
        self.__dict__.update(kwarg)
        self.name = 'c3synthetic-monkey'
        self.arms = list(range(self.L))
        self.x = {arm: suni(self.d) for arm in self.arms}
        self.theta = suni(self.d)
        self.oracle = argmax_oracle
        self.regret_avl = True
        logger.info(self)

    def __str__(self):
        return serialize(self, 'arms', 'x')

    def slot(self):
        self.xt = {arm: disturb(self.x[arm], self.v) for arm in self.arms}
        return self.xt

    def realize(self, action):
        return [pendulum() < self.theta.dot(self.xt[arm]) + self.b + np.random.normal(0, self.eps) for arm in action]
    
    def regret(self, action):
        Ew = {arm: self.theta.dot(self.xt[arm]) + self.b for arm in self.arms}
        opt = self.oracle(Ew, *self.params(True))
        p = [(self.theta.dot(self.xt[arm]) + self.b) / 2 + 0.5 for arm in action]
        popt = [(self.theta.dot(self.xt[arm]) + self.b) / 2 + 0.5 for arm in opt]
        return ereward(popt, self.gamma, self.disj) - ereward(p, self.gamma, self.disj)
    
    def params(self, descend):
        return (self.K, descend)

def contextual(s, cascade, rgamma, descend=None):
    logger.info('Initializing environment "{0}"| cascade:{1} rgamma:{2} descend:{3}'.format(s.name, cascade, rgamma, descend))
    def new():
        x = s.slot()
        p = s.params(descend)
        return x, p
    def play(action):
        ctr = s.realize(action)
        r, c = reward(ctr, s.gamma, s.disj)
        return r, c if cascade else [int(click) for click in ctr], s.regret(action)
    return ucb_settings(new=new, play=play, arms=s.arms, L=s.L, d=s.d, gamma=1-rgamma*(1-s.gamma), disj=s.disj, cascade=cascade, oracle=s.oracle, theta=s.theta, regret=s.regret_avl)
