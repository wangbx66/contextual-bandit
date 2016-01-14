import random

from scipy.linalg import sqrtm
import numpy as np
import colorama
colorama.init()

def red(s):
    return colorama.Fore.RED + s.__str__() + colorama.Fore.WHITE

def pendium():
    return np.random.uniform(-1, 1)

def uni(x):
    return x/np.sqrt(x.dot(x))

def suni(d):
    x = np.random.normal(0, 1, d)
    return uni(x)

def scat(d):
    e = uni(np.ones(d))
    x = [abs(suni(d)) for _ in range(int(15**(np.sqrt(d))))]
    x = [v for v in x if np.random.uniform() < v.dot(e)]
    v = random.sample(x, 1)[0]
    return v / v.sum()

def squni(d, linear=1):
    Z = np.random.uniform(0, 1, (d, d))
    V = Z.dot(np.linalg.inv(sqrtm(Z.T.dot(Z))))
    theta = suni(d)
    theta *= 1.145 * linear
    lamb = suni(d)
    alpha = lamb + V.dot(theta)
    if all(alpha > 0) or all(alpha < 0):
        return squni(d, linear)
    theta /= np.abs(alpha).max()
    lamb /= np.abs(alpha).max()
    Q = V.dot(np.diagflat(lamb)).dot(V.T)
    return theta, Q, lamb, V

def disturb(x, h):
    y = x + h * np.random.normal(0, 1, x.shape)
    return uni(y)

def reward(ctr, gamma, disj):
    conj = not disj
    for c, click in enumerate(ctr):
        if click == disj:
            return int(conj) + gamma ** c * (int(disj) - int(conj)), c
    return int(conj), len(ctr)

class ucb_settings:
    def __init__(self, **s):
        self.__dict__ = s

    def __str__(self):
        return '\n    '.join([str(k) + ' ' + str(v) for k, v in self.__dict__.items() if not k in ['arms', 'ctrh', 'users', 'U', 'V', 'x', 'A', 'G', 'oracle', 'theta']])

if __name__ == '__main__':
    theta, Q, lamb, V = squni(5)
    for _ in range(10000):
        x = suni(5)
        print(x.dot(Q).dot(x) + x.dot(theta), x.dot(Q).dot(x), x.dot(theta))
        assert(-1 < x.dot(Q).dot(x) + x.dot(theta) < 1)
