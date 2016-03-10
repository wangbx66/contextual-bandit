from functools import reduce
import random
import heapq

from scipy.linalg import sqrtm
import numpy as np
import colorama
colorama.init()

def red(s):
    return colorama.Fore.RED + s.__str__() + colorama.Fore.WHITE

def pendulum():
    return np.random.uniform(-1, 1)

def uni(x):
    return x/np.sqrt(x.dot(x))

def suni(d):
    x = np.random.normal(0, 1, d)
    return uni(x)

def scat(d):
    np.random.dirichlet((1,)*d)

def scat1(d):
    x = np.random.exponential(1, d)
    return x / x.sum()

def scat2(d):
    s = sorted(np.random.uniform(0, 1, d-1))
    return np.array([s[0], *[s[i+1] - s[i] for i in range(d-2)], 1-s[-1]])

def scat3(d):
    e = uni(np.ones(d))
    x = [abs(suni(d)) for _ in range(int(15**(np.sqrt(d))))]
    x = [v for v in x if np.random.uniform() < v.dot(e)]
    v = random.sample(x, 1)[0]
    return v / v.sum()

def squni(d, linear=0.6):
    Z = np.random.uniform(-1, 1, (d, d))
    V = Z.dot(np.linalg.inv(sqrtm(Z.T.dot(Z))))
    theta = suni(d)
    theta *= linear / np.abs(V.dot(theta)).max()
    lamb = suni(d)
    lamb *= (1 - linear) / np.abs(lamb).max()
    Q = V.dot(np.diagflat(lamb)).dot(V.T)
    return theta, Q, lamb, V

def disturb(x, h):
    y = x + h * np.random.normal(0, 1, x.shape)
    return uni(y)

def argmax_oracle(U, K, descend):
    return [p[1] for p in heapq.nlargest(K, [(U[arm], arm) for arm in U])][::2*descend-1]

def reward(ctr, gamma, disj):
    conj = not disj
    for c, click in enumerate(ctr):
        if click == disj:
            return int(conj) + gamma ** c * (int(disj) - int(conj)), c
    return int(conj), len(ctr)

def ereward(psarms, gamma, disj):
    if disj:
        return sum([gamma**i * reduce(float.__mul__, [1.,] + [1 - p for p in psarms][:i]) * psarms[i] for i in range(len(psarms))])
    else:
        return sum([(1 - gamma**i) * reduce(float.__mul__, [1.,] + psarms[:i]) * (1 - psarms[i]) for i in range(len(psarms))]) + reduce(float.__mul__, psarms)

def serialize(s, *blst):
    blst_local = blst + ('oracle', 'slot', 'realize', 'regret', 'params')
    return '\n    '.join([' '.join([str(k), str(v)]) for k, v in s.__dict__.items() if not k in blst_local])

class ucb_settings:
    def __init__(self, **s):
        self.__dict__ = s

    def __str__(self):
        return '\n    '.join([str(k) + ' ' + str(v) for k, v in self.__dict__.items() if len(str(v)) < 15])

if __name__ == '__main__':
    theta, Q, lamb, V = squni(5)
    for _ in range(10000):
        x = suni(5)
        print(x.dot(Q).dot(x) + x.dot(theta), x.dot(Q).dot(x), x.dot(theta))
        assert(-1 < x.dot(Q).dot(x) + x.dot(theta) < 1)
