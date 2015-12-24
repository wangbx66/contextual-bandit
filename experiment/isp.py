import logging
logger = logging.getLogger('ISP')

from utils import ucb_settings
from utils import suni

import networkx

def reachable(G, u, v):
    return networkx.has_path(G, u, v)

def isp_oracle(U, G, u, v):
    for (x, y), z in U.items():
        G[x][y]['weight'] = max(1 - z, 0)
    path = networkx.shortest_path(G, u, v, weight='weight')
    return [tuple(sorted((path[i], path[i+1]))) for i in range(len(path) - 1)]

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

def contextual_isp_rng(isp=1221, d=10, h=0.35, tlc=1, gamma=0.95, disj=False):
    G = isp_net(isp_data(isp))
    x = {e: suni(d) for e in G.edges()}
    theta = suni(d)
    logger.info('Initializing random settings "ISP" complete')
    s = ucb_settings(L=len(G.edges()), d=d, h=h, gamma=gamma, disj=disj, G=G, tlc=tlc, x=x, theta=theta)
    logger.info(s)
    return s
