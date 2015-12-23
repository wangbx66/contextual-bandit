import numpy as np
import colorama
colorama.init()

def red(s):
    return(colorama.Fore.RED + s.__str__() + colorama.Fore.WHITE)

def uni(x):
    return x/np.sqrt(x.dot(x))

def suni(d):
    x = np.abs(np.random.normal(0, 1, d))
    return uni(x)

def reward(ctr, gamma, disj):
    conj = not disj
    for c, click in enumerate(ctr):
        if click == disj:
            return int(conj) + gamma ** c * (int(disj) - int(conj)), c
    return int(conj), len(ctr)

def overlap(l1, l2):
    return len(set(l1) & l2)

class ucb_settings:
    def __init__(self, **s):
        self.__dict__ = s
        
    def __str__(self):
        return '\n    '.join([str(k) + ' ' + str(v) for k, v in self.__dict__.items() if not k in ['arms', 'ctrh', 'users', 'U', 'V', 'x', 'A']])
