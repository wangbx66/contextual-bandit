import logging
logger = logging.getLogger('Movielens')

from utils import ucb_settings
from utils import suni

import networkx

def isp_data(isp):
    assert isp in {1221,1239,1755,2914,3257,3356,3967,4755,6461,7018}
    for line in open('isp/{0}.cch'.format(isp)):
        for t in [x for x in line.strip().split() if x.startswith('<')]:
            yield int(line.strip().split()[0]), int(t[1:-1])

def isp_net(g):
    G = networkx.Graph()
    for s, t in g:
        G.add_edge(s, t)
    return G

def contextual_isp_rng(isp=1221, d=15, h=0.35, gamma=0.95, disj=True):
    G = isp_net(isp_data(isp))
    x = {e: suni(d) for e in G.edges()}
    theta = suni(d)
    logger.info('Initializing random settings "ISP" complete')
    s = ucb_settings(L=len(G.nodes()), d=d, gamma=gamma, disj=disj, G=G, x=x)
    logger.info(s)
    return s

g = isp_data(1221)
G = isp_net(g)
s = contextual_isp_rng()
