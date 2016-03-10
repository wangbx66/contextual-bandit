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
from agent import contextual_cascading_monkey
from agent import contextual_full_monkey
from agent import contextual_full_lijing
from agent import absolute_cascading_ucb
from environment import contextual_monkey_rng
from environment import contextual_monkey
from movielens import contextual_movielens_rng
from environment import contextual_movielens
from isp import contextual_isp_rng
from environment import contextual_isp

def flowtest_monkey(T, kw):
    logger.info('Monkey flowtest')
    logger.info('Require rng initialization')
    logger.info('\n    '.join(str(k) + ' ' + str(v) for k, v in kw.items()))
    s = contextual_monkey_rng(**kw)
    exploit1, explore1 = contextual_cascading_sherry(*contextual_monkey(s, cascade=True, rgamma=True, sort=False), T=T)
    exploit2, explore2 = contextual_cascading_sherry(*contextual_monkey(s, cascade=True, rgamma=False, sort=False), T=T)
    exploit3, explore3 = contextual_cascading_sherry(*contextual_monkey(s, cascade=True, rgamma=False, sort=True), T=T)
    # exploit2, explore2 = contextual_cascading_monkey(*contextual_monkey(s, cascade=True, rgamma=True, sort=False), T=T)
    exploit4, explore4 = contextual_full_lijing(*contextual_monkey(s, cascade=False, rgamma=False, sort=False), T=T)
    exploit5, explore5 = absolute_cascading_ucb(*contextual_monkey(s, cascade=True, rgamma=True, sort=False), T=T)
    exploit6, explore6 = absolute_cascading_ucb(*contextual_monkey(s, cascade=True, rgamma=False, sort=False), T=T)
    exploit7, explore7 = absolute_cascading_ucb(*contextual_monkey(s, cascade=True, rgamma=False, sort=True), T=T)
    # exploit6, explore6 = absolute_cascading_gammaucb(*contextual_monkey(s, cascade=True, rgamma=True, sort=False), T=T)
    # plt.plot(range(T), exploit1, 'or', range(T), exploit2, 'r--', range(T), exploit3, 'r', range(T), exploit4, 'y--', range(T), exploit5, 'b', range(T), exploit6, 'ob', range(T), exploit7, 'b--')
    plt.plot(range(T), exploit1, 'r--', range(T), exploit2, 'b--', range(T), exploit3, 'g--')

def flowtest_movielens(T, kw):
    logger.info('Movielens flowtest')
    logger.info('Require rng initialization')
    logger.info('\n    '.join(str(k) + ' ' + str(v) for k, v in kw.items()))
    s = contextual_movielens_rng(**kw)
    exploit1, explore1 = contextual_cascading_sherry(*contextual_movielens(s, cascade=True, rgamma=True, sort=False), T=T)
    # exploit2, explore2 = contextual_cascading_sherry(*contextual_movielens(s, cascade=True, rgamma=False, sort=False), T=T)
    exploit3, explore3 = contextual_cascading_sherry(*contextual_movielens(s, cascade=True, rgamma=False, sort=True), T=T)
    # exploit2, explore2 = contextual_cascading_monkey(*contextual_movielens(s, cascade=True, rgamma=True, sort=False), T=T)
    exploit4, explore4 = contextual_full_lijing(*contextual_movielens(s, cascade=False, rgamma=False, sort=False), T=T)
    exploit5, explore5 = absolute_cascading_ucb(*contextual_movielens(s, cascade=True, rgamma=True, sort=False), T=T)
    # exploit6, explore6 = absolute_cascading_ucb(*contextual_movielens(s, cascade=True, rgamma=False, sort=False), T=T)
    # exploit7, explore7 = absolute_cascading_ucb(*contextual_movielens(s, cascade=True, rgamma=False, sort=True), T=T)
    # plt.plot(range(T), exploit1, 'r--', range(T), exploit2, 'b--', range(T), exploit3, 'g--', range(T), exploit4, 'b--', range(T), exploit5, 'y--', range(T), exploit6, 'k--', range(T), exploit7, 'm--')
    plt.plot(range(T), exploit1, 'r--', range(T), exploit3, 'g--', range(T), exploit4, 'b--', range(T), exploit5, 'k--')
    #plt.plot(range(T), exploit1, 'r--', range(T), exploit2, 'b--', range(T), exploit3, 'g--')
    file_object = open('exploit1.txt', 'w')
    file_object.writelines(str(exploit1))
    file_object.close()
    file_object = open('exploit3.txt', 'w')
    file_object.writelines(str(exploit3))
    file_object.close()
    file_object = open('exploit4.txt', 'w')
    file_object.writelines(str(exploit4))
    file_object.close()
    file_object = open('exploit5.txt', 'w')
    file_object.writelines(str(exploit5))
    file_object.close()

def flowtest_isp(T, kw):
    logger.info('ISP flowtest')
    logger.info('Require rng initialization')
    logger.info('\n    '.join(str(k) + ' ' + str(v) for k, v in kw.items()))
    s = contextual_isp_rng(**kw)
    #exploit1, explore1 = contextual_cascading_monkey(*contextual_isp(s, cascade=True, rgamma=True), T=T)
    exploit1, explore1 = contextual_cascading_sherry(*contextual_isp(s, cascade=True, rgamma=True), T=T)
    # exploit2, explore2 = contextual_cascading_sherry(*contextual_isp(s, cascade=True, rgamma=False), T=T)
    exploit3, explore3 = contextual_full_lijing(*contextual_isp(s, cascade=False, rgamma=False), T=T)
    exploit4, explore4 = absolute_cascading_ucb(*contextual_isp(s, cascade=True, rgamma=True), T=T)
    # exploit5, explore5 = absolute_cascading_ucb(*contextual_isp(s, cascade=True, rgamma=False), T=T)
    # plt.plot(range(T), exploit1, 'r--', range(T), exploit2, 'b--', range(T), exploit3, 'g--', range(T), exploit4, 'k--')
    # plt.plot(range(T), exploit1, 'r--', range(T), exploit2, 'b--', range(T), exploit3, 'g--', range(T), exploit4, 'k--', range(T), exploit5, 'y--')
    plt.plot(range(T), exploit1, 'r--', range(T), exploit3, 'b--', range(T), exploit4, 'g--')
    file_object = open('exploit1.txt', 'w')
    file_object.writelines(str(exploit1))
    file_object.close()
    # file_object = open('exploit2.txt', 'w')
    # file_object.writelines(str(exploit2))
    # file_object.close()
    file_object = open('exploit3.txt', 'w')
    file_object.writelines(str(exploit3))
    file_object.close()
    file_object = open('exploit4.txt', 'w')
    file_object.writelines(str(exploit4))
    file_object.close()
    # file_object = open('exploit5.txt', 'w')
    # file_object.writelines(str(exploit5))
    # file_object.close()

#kw = {'L':20, 'd':10, 'h':0.35, 'K':4, 'gamma':0.9, 'eps':0.1, 'v':0.35, 'disj':True}
#kw = {'L':100, 'd':10, 'h':0.75, 'K':10, 'gamma':0.95, 'eps':0.1, 'v':0.35, 'disj':False}
#flowtest_monkey(T=3000, kw=kw)


# kw = {'L':200, 'portion':0.2, 'd':5, 'K':4, 'h':60, 'gamma':0.95, 'disj':True}
# flowtest_movielens(T=1000, kw=kw)

kw = {'isp':1755, 'd':15, 'h':0.35, 'tlc':0.75, 'gamma':0.9, 'disj':False}
flowtest_isp(T=300000, kw=kw)
plt.show()
logfile.close()
