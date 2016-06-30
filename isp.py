from functools import reduce
from itertools import islice
import random
import logging
logger = logging.getLogger('ISP')

import networkx

from utils import suni
from utils import disturb
from utils import serialize
from utils import pendulum
from utils import ereward

def reachable(G, u, v):
    return networkx.has_path(G, u, v)

def isp_oracle(U, G, u, v):
    for (x, y), z in U.items():
        G[x][y]['weight'] = max(1 - z, 0)
    path = networkx.shortest_path(G, u, v, weight='weight')
    return [tuple(sorted((path[i], path[i+1]))) for i in range(len(path) - 1)]

def isp_Zoracle(U, G, u, v, k, gamma):
    for (x, y), z in U.items():
        G[x][y]['weight'] = max(1 - z, 0)
    s = {}
    for path in islice(networkx.shortest_simple_paths(G, u, v), k):
        sarms = tuple(tuple(sorted((path[i], path[i+1]))) for i in range(len(path) - 1))
        psarms = [(U[e] + 1) / 2 for e in sarms]
        s[sarms] = ereward(psarms, gamma, False)
    return list(max(s, key=s.get))

def isp_Coracle(U, G, u, v, k, gamma):
    for (x, y), _ in U.items():
        G[x][y]['weight'] = max(1, 0)
    s = {}
    for path in islice(networkx.shortest_simple_paths(G, u, v), k):
        sarms = tuple(tuple(sorted((path[i], path[i+1]))) for i in range(len(path) - 1))
        psarms = [(U[e] + 1) / 2 for e in sarms]
        s[sarms] = ereward(psarms, gamma, False)
    return list(max(s, key=s.get))

def isp_data(isp):
    assert isp in {1221,1239,1755,2914,3257,3356,3967,4755,6461,7018}
    for line in open('isp/{0}.cch'.format(isp)):
        for t in [x for x in line.strip().split() if x.startswith('<')]:
            yield int(line.strip().split()[0]), int(t[1:-1])

def isp_net(g):
    G = networkx.Graph()
    for u, v in g:
        G.add_edge(u, v, weight=1)
    return G

class c3synthetic_isp_rng:

    # csynthetic_isp_rng(isp=1221, d=10, v=0.35, k=400, gamma=0.95)

    def __init__(self, **kwarg):
        logger.info('Initializing random settings "Contextual Synthetic ISP"')
        self.__dict__.update(kwarg)
        self.disj = False
        self.name = 'c3synthetic-isp'
        self.G = isp_net(isp_data(self.isp))
        self.arms = self.G.edges()
        self.x = {e: suni(self.d) for e in self.G.edges()}
        self.theta = suni(self.d)
        self.L = len(self.G.edges())
        self.regret_avl = True
        self.load_oracle()
        logger.info(self)

    def __str__(self):
        return serialize(self, 'arms', 'x', 'G', 'reachable')

    def load_oracle(self):
        self.oracle = isp_oracle

    def reachable(self):
        return networkx.has_path(self.G, self.u, self.v)

    def slot(self):
        self.xt = {arm: disturb(self.x[arm], self.v) for arm in self.arms}
        self.u, self.v = random.sample(self.G.nodes(), 2)
        return self.xt if self.reachable() else self.slot()

    def realize(self, action):
        return [pendulum() < self.theta.dot(self.xt[arm]) for arm in action]

    def regret(self, action):
        Ew = {arm: self.theta.dot(self.xt[arm]) for arm in self.arms}
        opt = self.oracle(Ew, *self.params(None))
        p = [self.theta.dot(self.xt[arm]) / 2 + 0.5 for arm in action]
        popt = [self.theta.dot(self.xt[arm]) / 2 + 0.5 for arm in opt]
        regret = ereward(popt, self.gamma, self.disj) - ereward(p, self.gamma, self.disj)
        return ereward(popt, self.gamma, self.disj) - ereward(p, self.gamma, self.disj)

    def params(self, descend):
        return (self.G, self.u, self.v)

class c3synthetic_Zisp_rng(c3synthetic_isp_rng):
    def load_oracle(self):
        self.oracle = isp_Zoracle

    def params(self, descend):
        return (self.G, self.u, self.v, self.k, self.gamma)

class c3synthetic_Cisp_rng(c3synthetic_isp_rng):
    def load_oracle(self):
        self.oracle = isp_Coracle

    def params(self, descend):
        return (self.G, self.u, self.v, self.k, self.gamma)
