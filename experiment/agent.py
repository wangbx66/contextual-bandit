import time
import logging
logger = logging.getLogger('Agent')

import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def contextual_cascading_monkey(s, T):
    reward = [0]
    regret = [0]
    for t in range(1, T):
        _, params = s.new()
        U = {arm: np.random.uniform(0, 1) for arm in s.arms}
        recc = s.oracle(U, *params)
        r, _, z = s.play(recc)
        reward.append(reward[-1] + r)
        regret.append(regret[-1] + z)
    logger.info('Monkey play reward {0}/{1}'.format(reward[-1], T))
    if s.regret is True:
        logger.info('regret {0}/{1}'.format(regret[-1], T))
    return reward, regret, None

contextual_full_monkey = contextual_cascading_monkey

def absolute_cascading_ucb(s, T):
    reward = [0]
    regret = [0]
    W = {arm: 0.5 for arm in s.arms}
    b = {arm: 1 for arm in s.arms}
    for t in range(1, T):
        _, params = s.new()
        U = {arm: W[arm] + np.sqrt(1.5 * np.log(t) / b[arm]) for arm in s.arms}
        recc = s.oracle(U, *params)
        r, c, z = s.play(recc)
        for k in range(min(len(recc), c+1)):
            W[recc[k]] = ((b[recc[k]]) * W[recc[k]] + s.gamma ** k * ((k == c) == s.disj)) / (b[recc[k]] + s.gamma ** k)
            b[recc[k]] += s.gamma ** k
        reward.append(reward[-1] + r)
        regret.append(regret[-1] + z)
    logger.info('Absolute play reward {0}/{1}'.format(reward[-1], T))
    if s.regret is True:
        logger.info('regret {0}/{1}'.format(regret[-1], T))
    return reward, regret, None

def contextual_cascading_sherry(s, T, delta=0.9, lamb=0.1):
    assert s.cascade
    theta = np.zeros(s.d)
    beta = 0
    V = lamb * np.eye(s.d)
    ldV = np.linalg.slogdet(V)[1]
    X = np.zeros((1, s.d))
    Y = np.zeros(1)
    reward = [0]
    regret = [0]
    timestamp = time.time()
    for t in range(1, T):
        print(t)
        x, params = s.new()
        U = {arm: theta.dot(x[arm]) + beta * x[arm].dot(np.linalg.inv(V)).dot(x[arm]) for arm in x}
        recc = s.oracle(U, *params)
        r, c, z = s.play(recc)
        V += sum([s.gamma ** (2*k) * np.outer(x[recc[k]], x[recc[k]]) for k in range(min(len(recc), c+1))])
        X = np.concatenate([X] + [s.gamma ** k * x[recc[k]].reshape(1, s.d) for k in range(min(len(recc), c+1))])
        Y = np.concatenate([Y] + [s.gamma ** k * ((k == c) == s.disj) * np.ones(1) for k in range(min(len(recc), c+1))])
        theta = np.linalg.inv(X.T.dot(X) + lamb * np.eye(s.d)).dot(X.T.dot(Y))
        beta = np.sqrt(np.linalg.slogdet(V)[1] - ldV - 2 * np.log(delta)) + np.sqrt(lamb)
        reward.append(reward[-1] + r)
        regret.append(regret[-1] + z)
    logger.info('Sherry play reward {0}/{1}'.format(reward[-1], T))
    if s.theta is not None:
        logger.info('theta cosine similarity {0}'.format(1 - cosine(s.theta, theta)))
        similarity = 1 - cosine(s.theta, theta)
    else:
        similarity = None
    if s.regret is True:
        logger.info('regret {0}/{1}'.format(regret[-1], T))
    return reward, regret, similarity

def contextual_cascading_gsherry(s, T, delta=0.9, lamb=0.1, gamma=None):
    assert gamma
    assert s.cascade
    theta = np.zeros(s.d)
    beta = 0
    V = lamb * np.eye(s.d)
    ldV = np.linalg.slogdet(V)[1]
    X = np.zeros((1, s.d))
    Y = np.zeros(1)
    reward = [0]
    regret = [0]
    timestamp = time.time()
    for t in range(1, T):
        x, params = s.new()
        U = {arm: theta.dot(x[arm]) + beta * x[arm].dot(np.linalg.inv(V)).dot(x[arm]) for arm in x}
        recc = s.oracle(U, *params[:-1], gamma)
        r, c, z = s.play(recc)
        V += sum([gamma ** (2*k) * np.outer(x[recc[k]], x[recc[k]]) for k in range(min(len(recc), c+1))])
        X = np.concatenate([X] + [gamma ** k * x[recc[k]].reshape(1, s.d) for k in range(min(len(recc), c+1))])
        Y = np.concatenate([Y] + [gamma ** k * ((k == c) == s.disj) * np.ones(1) for k in range(min(len(recc), c+1))])
        theta = np.linalg.inv(X.T.dot(X) + lamb * np.eye(s.d)).dot(X.T.dot(Y))
        beta = np.sqrt(np.linalg.slogdet(V)[1] - ldV - 2 * np.log(delta)) + np.sqrt(lamb)
        reward.append(reward[-1] + r)
        regret.append(regret[-1] + z)
    logger.info('Sherry play reward {0}/{1}, gamma={2}'.format(reward[-1], T, gamma))
    if s.theta is not None:
        logger.info('theta cosine similarity {0}'.format(1 - cosine(s.theta, theta)))
        similarity = 1 - cosine(s.theta, theta)
    else:
        similarity = None
    if s.regret is True:
        logger.info('regret {0}/{1}'.format(regret[-1], T))
    return reward, regret, similarity

def contextual_full_lijing(s, T, delta=0.9, lamb=0.1):
    assert not s.cascade
    theta = np.zeros(s.d)
    beta = 0
    V = lamb * np.eye(s.d)
    b = np.zeros(s.d)
    reward = [0]
    regret = [0]
    for t in range (1, T):
        x, params = s.new()
        U = {arm: theta.dot(x[arm]) + beta * x[arm].dot(np.linalg.inv(V)).dot(x[arm]) for arm in x}
        recc = s.oracle(U, *params)
        r, ctr, z = s.play(recc)
        V += sum([(s.gamma)**(2*k) * np.outer(x[recc[k]], x[recc[k]]) for k in range(len(recc))])
        b += sum([ctr[k] * x[recc[k]] for k in range(len(recc))])
        theta = np.linalg.inv(V).dot(b)
        beta = np.sqrt(s.d * np.log((1 + t*s.L) / delta)) + np.sqrt(lamb)
        reward.append(reward[-1] + r)
        regret.append(regret[-1] + z)
    logger.info('Lijing play reward {0}/{1}'.format(reward[-1], T))
    if s.theta is not None:
        logger.info('theta cosine similarity {0}'.format(1 - cosine(s.theta, theta)))
        similarity = 1 - cosine(s.theta, theta)
    else:
        similarity = None
    if s.regret is True:
        logger.info('regret {0}/{1}'.format(regret[-1], T))
    return reward, regret, similarity
