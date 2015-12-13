import numpy as np
import colorama
colorama.init()

def red(s):
    return(colorama.Fore.RED + s.__str__() + colorama.Fore.WHITE)

def uni(x):
    return x/np.sqrt(x.dot(x))

def reward(ctr, gamma, disj):
    for c, click in enumerate(ctr):
        if click ^ disj:
            return int(disj) + gamma ** c * (int(not disj) - int(disj)), c
    return int(disj), len(ctr)

def overlap(l1, l2):
    return len(set(l1) & l2)

class ucb_settings:
    def __init__(self, **s):
        self.__dict__ = s
        
    def __str__(self):
        return '\n'.join([str(k) + ' ' + str(v) for k, v in self.__dict__.items() if not k in ['arms', 'ctrh', 'users', 'U', 'V', 'x']])
