'''
Copyright 2019, INRIA
CVT Uility functions based on pymap_elites framework https://github.com/resibots/pymap_elites/blob/master/map_elites/
pymap_elites main contributor(s):
    Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
    Eloise Dalin , eloise.dalin@inria.fr
    Pierre Desreumaux , pierre.desreumaux@inria.fr
Modified by Olle Nilsson: olle.nilsson19@imperial.ac.uk
'''

import numpy as np

from pathlib import Path
from sklearn.cluster import KMeans


def cvt(k, dim, samples, cvt_use_cache=True):
    # check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            if dim == 1:
                if k == 1:
                    return np.expand_dims(np.expand_dims(np.loadtxt(fname), axis=0), axis=1)
                return np.expand_dims(np.loadtxt(fname), axis=1)
            else:
                if k == 1:
                    return np.expand_dims(np.loadtxt(fname), axis=0)
                return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)
    x = np.random.rand(samples, dim)

    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, max_iter=1000000, verbose=1, tol=1e-8) #Full is the proper Expectation Maximization algorithm
    k_means.fit(x)
    write_centroids(k_means.cluster_centers_)
    return k_means.cluster_centers_


def __centroids_filename(k, dim):
    return 'my_CVT/centroids_' + str(k) + '_' + str(dim) + '.dat'


def write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')

def make_hashable(array):
    return tuple(map(float, array))


class Individual:
    def __init__(self, x, desc, fitness, centroid=None):  # x是actor
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid
        self.novelty = None


def add_to_archive(individuals, archive, kdt, main=True):
    for i in individuals:
        fitness = i.fitness
        desc = i.desc
        niche_index = kdt.query([desc], k=1)[1][0][0]
        niche = kdt.data[niche_index]
        n = make_hashable(niche)
        if main:
            i.centroid = n
        if n in archive:
            if fitness > archive[n].fitness:
                archive[n] = i
        else:
            archive[n] = i

    # niche_index = kdt.query([centroid], k=1)[1][0][0]
    # niche = kdt.data[niche_index]
    # n = make_hashable(niche)
    # if main:
    #     s.centroid = n
    # if n in archive:
    #     if s.fitness > archive[n].fitness:
    #         if main:
    #             s.x.novel = False
    #             s.x.delta_f = s.fitness - archive[n].fitness
    #         archive[n] = s
    #         return 1
    #     return 0
    # else:
    #     archive[n] = s
    #     if main:
    #         s.x.novel = True
    #         s.x.delta_f = None
    #     return 1
