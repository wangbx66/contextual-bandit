import logging
logging.basicConfig(format='%(name)s-%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger('Main')

import time
seed = int(time.time() * 100) % 339
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
logger.info('Numpy/Python random seed = {0}'.format(seed))

import matplotlib.pyplot as plt

from agent import contextual_cascading_sherry
from agent import contextual_cascading_monkey
from agent import contextual_full_monkey
from agent import contextual_full_lijing
from agent import absolute_cascading_ucb
from environment import contextual_monkey_rng
from environment import contextual_cascading_monkey as contextual_cascading_monkey_environment
from environment import contextual_full_monkey as contextual_full_monkey_environment
from movielens import contextual_movielens_rng
from environment import contextual_cascading_movielens

def flowtest_monkey_disj():
    T = 10000
    s = contextual_monkey_rng(L=20, d=10, h=0.75, K=4, gamma=0.95, eps=0.1, v=0.35, disj=True)
    exploit1, explore1 = contextual_cascading_sherry(*contextual_cascading_monkey_environment(s), T=T)
    exploit2, explore2 = contextual_cascading_monkey(*contextual_cascading_monkey_environment(s), T=T)
    exploit3, explore3 = contextual_full_monkey(*contextual_full_monkey_environment(s), T=T)
    exploit4, explore4 = contextual_full_lijing(*contextual_full_monkey_environment(s), T=T)
    exploit5, explore5 = absolute_cascading_ucb(*contextual_cascading_monkey_environment(s), T=T)
    plt.plot(range(T), exploit1, 'r--', range(T), exploit2, 'r--', range(T), exploit4, 'b--')

def flowtest_monkey_conj():
    T = 10000
    s = contextual_monkey_rng(L=20, d=10, h=0.35, K=4, gamma=0.95, eps=0.1, v=0.35, disj=False)
    exploit1, explore1 = contextual_cascading_sherry(*contextual_cascading_monkey_environment(s), T=T)
    exploit2, explore2 = contextual_cascading_monkey(*contextual_cascading_monkey_environment(s), T=T)
    exploit3, explore3 = contextual_full_monkey(*contextual_full_monkey_environment(s), T=T)
    exploit4, explore4 = contextual_full_lijing(*contextual_full_monkey_environment(s), T=T)
    exploit5, explore5 = absolute_cascading_ucb(*contextual_cascading_monkey_environment(s), T=T)
    plt.plot(range(T), exploit1, 'r--', range(T), exploit2, 'r--', range(T), exploit4, 'b--')

T = 10000
s = contextual_movielens_rng(L=None, portion=0.5, d=15, K=8, h=13500, gamma=0.95, disj=False)
exploit1, explore1 = contextual_cascading_monkey(*contextual_cascading_movielens(s), T=T)
exploit2, explore2 = contextual_cascading_sherry(*contextual_cascading_movielens(s), T=T)
