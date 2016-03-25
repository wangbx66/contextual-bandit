import logging
logfile = logging.FileHandler('log')
logfile.setLevel(logging.INFO)
logconsole = logging.StreamHandler()
logconsole.setLevel(logging.DEBUG)
logging.basicConfig(format='%(name)s-%(levelname)s: %(message)s', level=logging.DEBUG, handlers=[logfile, logconsole])
logger = logging.getLogger('Main')

import time
seed = int(time.time() * 100) % 339
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
logger.info('===== Numpy/Python random seed = {0}'.format(seed))

import matplotlib.pyplot as plt

from agent import contextual_cascading_sherry
from agent import contextual_cascading_gsherry
from agent import contextual_cascading_monkey
from agent import contextual_full_monkey
from agent import contextual_full_lijing
from agent import absolute_cascading_ucb
from environment import contextual
from environment import c3synthetic_monkey_rng
from movielens import c3_movielens_rng
from isp import c3synthetic_isp_rng
from isp import c3synthetic_Zisp_rng
from isp import c3synthetic_Cisp_rng

def flowtest_monkey(T, **kw):
    logger.info('Monkey flowtest')
    logger.info('Require rng initialization')
    logger.info('\n    '.join(str(k) + ' ' + str(v) for k, v in kw.items()))
    s = c3synthetic_monkey_rng(**kw)
    reward, regret, similarity = contextual_cascading_sherry(contextual(s, cascade=True, rgamma=True, descend=False), T=T)
    reward, regret, similarity = contextual_cascading_sherry(contextual(s, cascade=True, rgamma=False, descend=False), T=T)
    reward, regret, similarity = contextual_cascading_sherry(contextual(s, cascade=True, rgamma=False, descend=True), T=T)
    reward, regret, similarity = contextual_full_lijing(contextual(s, cascade=False, rgamma=False, descend=False), T=T)
    reward, regret, similarity = absolute_cascading_ucb(contextual(s, cascade=True, rgamma=True, descend=False), T=T)
    reward, regret, similarity = absolute_cascading_ucb(contextual(s, cascade=True, rgamma=False, descend=False), T=T)
    reward, regret, similarity = absolute_cascading_ucb(contextual(s, cascade=True, rgamma=False, descend=True), T=T)

def flowtest_movielens(T, **kw):
    logger.info('Movielens flowtest')
    logger.info('Require rng initialization')
    logger.info('\n    '.join(str(k) + ' ' + str(v) for k, v in kw.items()))
    s = c3_movielens_rng(**kw)
    reward, regret, similarity = contextual_cascading_sherry(contextual(s, cascade=True, rgamma=True, descend=False), T=T)
    reward, regret, similarity = contextual_cascading_sherry(contextual(s, cascade=True, rgamma=True, descend=True), T=T)
    reward, regret, similarity = contextual_full_lijing(contextual(s, cascade=False, rgamma=True, descend=False), T=T)
    reward, regret, similarity = absolute_cascading_ucb(contextual(s, cascade=True, rgamma=True, descend=False), T=T)
    reward, regret, similarity = absolute_cascading_ucb(contextual(s, cascade=True, rgamma=True, descend=True), T=T)

def flowtest_isp(T, **kw):
    logger.info('ISP flowtest')
    logger.info('Require rng initialization')
    logger.info('\n    '.join(str(k) + ' ' + str(v) for k, v in kw.items()))
    s = c3synthetic_isp_rng(**kw)
    reward, regret, similarity = contextual_cascading_sherry(contextual(s, cascade=True, rgamma=True), T=T)
    reward, regret, similarity = contextual_cascading_sherry(contextual(s, cascade=True, rgamma=False), T=T)
    reward, regret, similarity = contextual_full_lijing(contextual(s, cascade=False, rgamma=False), T=T)
    reward, regret, similarity = absolute_cascading_ucb(contextual(s, cascade=True, rgamma=True), T=T)
    reward, regret, similarity = absolute_cascading_ucb(contextual(s, cascade=True, rgamma=False), T=T)

def flowtest_gisp(T, **kw):
    logger.info('Gamma ISP flowtest')
    logger.info('Require rng initialization')
    logger.info('\n    '.join(str(k) + ' ' + str(v) for k, v in kw.items()))
    s = c3synthetic_Zisp_rng(**kw)
    reward, regret, similarity = contextual_cascading_gsherry(contextual(s, cascade=True, rgamma=True), T=T, gamma=1.00)
    reward, regret, similarity = contextual_cascading_gsherry(contextual(s, cascade=True, rgamma=True), T=T, gamma=0.98)
    reward, regret, similarity = contextual_cascading_gsherry(contextual(s, cascade=True, rgamma=True), T=T, gamma=0.96)
    reward, regret, similarity = contextual_cascading_gsherry(contextual(s, cascade=True, rgamma=True), T=T, gamma=0.94)
    reward, regret, similarity = contextual_cascading_gsherry(contextual(s, cascade=True, rgamma=True), T=T, gamma=0.92)
    reward, regret, similarity = contextual_cascading_gsherry(contextual(s, cascade=True, rgamma=True), T=T, gamma=0.9)
    reward, regret, similarity = contextual_cascading_gsherry(contextual(s, cascade=True, rgamma=True), T=T, gamma=0.88)
    reward, regret, similarity = contextual_cascading_gsherry(contextual(s, cascade=True, rgamma=True), T=T, gamma=0.86)
    reward, regret, similarity = contextual_cascading_gsherry(contextual(s, cascade=True, rgamma=True), T=T, gamma=0.84)
    reward, regret, similarity = contextual_cascading_gsherry(contextual(s, cascade=True, rgamma=True), T=T, gamma=0.82)
    reward, regret, similarity = contextual_cascading_gsherry(contextual(s, cascade=True, rgamma=True), T=T, gamma=0.80)

#kw = {'L':20, 'd':5, 'b':0, 'K':4, 'gamma':0.9, 'eps':0.1, 'v':0.35, 'disj':True}
#kw = {'L':100, 'd':10, 'b':0, 'K':10, 'gamma':0.95, 'eps':0.1, 'v':0.35, 'disj':False}
kw = {'L':1000, 'd':500, 'b':0, 'K':10, 'gamma':0.95, 'eps':0.1, 'v':0.35, 'disj':False}
flowtest_monkey(500, **kw)

#kw = {'n_movies':20, 'train_portion':0.7, 'd':3, 'K':4, 'n_users':1500, 'gamma':1.00, 'disj':True}
#flowtest_movielens(1500, **kw)

#kw = {'isp':6461, 'd':5, 'v':0.35, 'k':10, 'gamma':0.90}
#flowtest_isp(1000, **kw)

#kw = {'isp':6461, 'd':5, 'v':0.35, 'k':10, 'gamma':0.90}
#flowtest_gisp(1000, **kw)

#plt.show()
logfile.close()
