import math
import sqlite3
import sklearn
import numpy as np
import pdb

import scipy.cluster
import multiprocessing as mp
import itertools as it
import time
import pickle

from sklearn.cluster import KMeans
#from util import Canonicalizer
from collections import *
import random
import prettyplotlib as ppl

# This is "import matplotlib.pyplot as plt" from the prettyplotlib library
from prettyplotlib import plt

# This is "import matplotlib as mpl" from the prettyplotlib library
from prettyplotlib import mpl

from microscopes.models import gp as gamma_poisson, dm as dirichlet_multinomial
from microscopes.mixture.definition import model_definition
from microscopes.cxx.common.rng import rng
from microscopes.cxx.common.recarray.dataview import numpy_dataview
from microscopes.cxx.common.scalar_functions import log_exponential
from microscopes.cxx.mixture.model import initialize, bind, deserialize
from microscopes.cxx.kernels import gibbs, slice

def algo(counts):
    """
    receives a [word x year] matrix of non-negative counts.

    expected to return a list of assignment vectors
    """

def smooth(counts):
    return np.array([[np.mean(c[i:i+3]) for i in xrange(len(c))] for c in counts])

def smooth_ints(counts):
    return np.round(smooth(counts)).astype(int)

def scores(values):
    return np.array([l / max(l) for l in values])

def take_last_merge_strategy(assignments):
    return assignments[-1]

def kmeans_algo(nclusters):
    def algo(counts):
        data = scores(smooth(counts))
        clusterer = KMeans(nclusters, n_init=30, init='k-means++')
        clusterer.fit(data)
        return [clusterer.labels_]
    return algo

# multiprocessing boilerplate we will get rid of eventually

def _make_definition_gamma_poisson(N, D):
    return model_definition(N, [gamma_poisson]*D)

def _make_hparams_gamma_poisson(D):
  hparams = {
    y : {
      'alpha': (log_exponential(1.), 1.),
      'inv_beta': (log_exponential(0.1), 0.1),
    }
    for y in xrange(D)
  }
  return hparams

def _revive_gamma_poisson(N, D, data, latent):
    defn = _make_definition_gamma_poisson(N, D)
    view = numpy_dataview(data)
    latent = deserialize(defn, latent)
    hparams = _make_hparams_gamma_poisson(D)
    return defn, view, latent, hparams

def _work_gamma_poisson(args):
    (N, D), data, latent, iters = args
    defn, view, latent, hparams = _revive_gamma_poisson(N, D, data, latent)
    r = rng()
    model = bind(latent, view)
    for _ in xrange(iters):
        gibbs.assign(model, r)
        #slice.hp(model, r, hparams=hparams)
    return latent.serialize()

def _make_definition_dirichlet_multinomial(N, D):
    return model_definition(N, [dirichlet_multinomial(D)])

def _revive_dirichlet_multinomial(N, D, data, latent):
    defn = _make_definition_dirichlet_multinomial(N, D)
    view = numpy_dataview(data)
    latent = deserialize(defn, latent)
    return defn, view, latent

def _work_dirichlet_multinomial(args):
    (N, D), data, latent, iters = args
    defn, view, latent = _revive_dirichlet_multinomial(N, D, data, latent)
    r = rng()
    model = bind(latent, view)
    for _ in xrange(iters):
        gibbs.assign(model, r)
    return latent.serialize()

def mixturemodel_gamma_poisson(nchains=100, nitersperchain=1000, pickle_file='results.p'):
    def algo(counts):
        data = smooth_ints(counts)
        N, D = data.shape
        latents = []
        defn = _make_definition_gamma_poisson(N, D)
        Y = np.array([tuple(y) for y in data], dtype=[('',np.int32)]*D)
        view = numpy_dataview(Y)
        r = rng()
        for _ in xrange(nchains):
            latent = initialize(defn, view, r=r)
            latents.append(latent.serialize())
        p = mp.Pool(processes=mp.cpu_count() * 2)
        start_time = time.time()
        infers = p.map_async(_work_gamma_poisson, [((N, D), Y, latent, nitersperchain) for latent in latents]).get(10000000)
        print "inference took", (time.time() - start_time), "secs"
        p.close()
        p.join()
        with open(pickle_file, 'w') as fp:
            pickle.dump(list(infers), fp)
        infers = [_revive_gamma_poisson(N, D, Y, infer)[2] for infer in infers]
        assignments_samples = [s.assignments() for s in infers]
        return assignments_samples
    return algo

def mixturemodel_dirichlet_multinomial(nchains=100, nitersperchain=1000, pickle_file='results.p'):
    def algo(counts):
        data = smooth_ints(counts)
        N, D = data.shape
        latents = []
        defn = _make_definition_dirichlet_multinomial(N, D)
        Y = np.array([(list(y),) for y in data], dtype=[('',np.int32, (D,))])
        view = numpy_dataview(Y)
        r = rng()
        for _ in xrange(nchains):
            latent = initialize(defn, view, r=r)
            latents.append(latent.serialize())
        p = mp.Pool(processes=mp.cpu_count() * 2)
        start_time = time.time()
        infers = p.map_async(_work_dirichlet_multinomial, [((N, D), Y, latent, nitersperchain) for latent in latents]).get(10000000)
        print "inference took", (time.time() - start_time), "secs"
        p.close()
        p.join()
        with open(pickle_file, 'w') as fp:
            pickle.dump(list(infers), fp)
        infers = [_revive_dirichlet_multinomial(N, D, Y, infer)[2] for infer in infers]
        assignments_samples = [s.assignments() for s in infers]
        return assignments_samples
    return algo

def scipy_fcluster_merge_strategy(zmat_file_prefix='zmat'):
    def merge(assignments):
        # Z-matrix helpers
        def groups(assignments):
            cluster_map = {}
            for idx, gid in enumerate(assignments):
                lst = cluster_map.get(gid, [])
                lst.append(idx)
                cluster_map[gid] = lst
            return tuple(sorted(map(tuple, cluster_map.values()), key=len, reverse=True))

        def zmatrix(assignments_samples):
            n = len(assignments_samples[0])
            # should make sparse matrix
            zmat = np.zeros((n, n), dtype=np.float32)
            for assignments in assignments_samples:
                clusters = groups(assignments)
                for cluster in clusters:
                    for i, j in it.product(cluster, repeat=2):
                        zmat[i, j] += 1
            zmat /= float(len(assignments_samples))
            return zmat

        def reorder_zmat(zmat, order):
            zmat = zmat[order]
            zmat = zmat[:,order]
            return zmat

        def linkage(zmat):
            assert zmat.shape[0] == zmat.shape[1]
            zmat0 = np.array(zmat[np.triu_indices(zmat.shape[0], k=1)])
            zmat0 = 1. - zmat0
            return scipy.cluster.hierarchy.linkage(zmat0)

        def ordering(l):
            return np.array(scipy.cluster.hierarchy.leaves_list(l))

        zmat = zmatrix(assignments)
        li = linkage(zmat)

        # diagnostic: draw z-matrix
        indices = ordering(li)
        plt.imshow(reorder_zmat(zmat, indices),
               cmap=plt.cm.binary, interpolation='nearest')
        plt.savefig("{}-before.pdf".format(zmat_file_prefix))
        plt.close()

        fassignment = scipy.cluster.hierarchy.fcluster(li, 0.001)
        clustering = groups(fassignment)
        sorted_ordering = list(it.chain.from_iterable(clustering))

        # draw post fcluster() ordered z-matrix
        plt.imshow(reorder_zmat(zmat, sorted_ordering),
               cmap=plt.cm.binary, interpolation='nearest')
        plt.savefig("{}-after.pdf".format(zmat_file_prefix))
        plt.close()
        #return fassignment
        return np.array(assignments[0])

    return merge

def unzip(zipped):
    a, b = [], []
    for x, y in zipped:
        a.append(x)
        b.append(y)
    return a, b

def cluster_and_render(conf,
                       dbname,
                       algo_fn=kmeans_algo(8),
                       merge_fn=take_last_merge_strategy,
                       outname="./text.html"):

  db = sqlite3.connect(dbname)
  r = db.execute("select min(year), max(year) from counts")
  minyear, maxyear = r.fetchone()

  def vectors():
    r = db.execute("select word, year, c from counts order by word, year")
    vects = defaultdict(dict)
    for w,y,c in r:
      l = vects[w]
      l[y] = c

    ret = []
    words = []
    for w in vects:
      d = vects[w]
      counts = [d.get(y, 0) for y in xrange(minyear, maxyear+1)]
      ret.append(counts)
      words.append(w)
    return words, np.array(ret)

  words, counts = vectors()
  # exclude words which have culmulative count < filter
  words, counts = unzip([(w, c) for w, c in zip(words, counts) if sum(c) >= 5])
  counts = np.array(counts)

  print 'len(words):', len(words)

  labels = merge_fn(algo_fn(counts))
  vects = smooth(counts)
  vects = np.array([[w] + list(v) for w, v in zip(words, vects)])

  xs = np.array(range(minyear, maxyear+1))

  imgs = []
  content = []

  def add_content(subcluster, content, suffix):
      fig, ax = plt.subplots(1, figsize=(6.5,2.5))
      for childax in ax.get_children():
        if isinstance(childax, mpl.spines.Spine):
          childax.set_color('#aaaaaa')
      for i in ax.get_xticklabels():
        i.set_color('#aaaaaa')
      for i in ax.get_yticklabels():
        i.set_color('#aaaaaa')

      subcluster = sorted(subcluster, key=lambda t: max(t[1:].astype(float)), reverse=True)[:10]
      subcluster = np.array(subcluster)
      words = subcluster[:,0]
      ys = subcluster[:,1:].astype(float)
      mean = [np.mean(ys[:,i]) for i in xrange(ys.shape[1])]
      ys = ys.transpose()
      ax.set_ylim(top=max(10, max(map(max, ys))))

      ppl.plot(ax, xs, ys, alpha=0.3, color="#7777ee")
      ppl.plot(ax, xs, mean, alpha=1, color="black")
      fname = './plots/%s_%s.png' % (conf, suffix)
      fig.savefig(fname, format='png')

      maxes = map(max, ys)
      idx = maxes.index(max(maxes))
      content.append(('', words, fname, idx))


  for label in set(labels):
    fig, ax = plt.subplots(1, figsize=(13, 5))

    idxs = labels == label
    cluster = vects[idxs]
    cluster = sorted(cluster, key=lambda t: max(t[1:].astype(float)), reverse=True)
    cluster = filter(lambda l: sum(map(float, l[1:])) > 4, cluster)
    if len(cluster) < 10: continue
    cluster = np.array(cluster)
    words = cluster[:,0]
    words = list(words)

    cluster = cluster[:10]
    add_content(cluster, content, label)

  content.sort(key=lambda c: c[-1])

  from jinja2 import Template
  template = Template(file('./clustertemplate.html').read())


  with file(outname, 'w') as f:
    f.write( template.render(content=content))

if __name__ == '__main__':

  cluster_and_render('sigmod_kmeans', 'stats.db', outname="./text_kmeans.html")

  #cluster_and_render('sigmod_mixturemodel_gp', 'stats.db',
  #        algo_fn=mixturemodel_gamma_poisson(pickle_file='states_mixturemodel_gp.p'),
  #        merge_fn=scipy_fcluster_merge_strategy(zmat_file_prefix='zmat_mixturemodel_gp'),
  #        outname="./text_mixturemodel_gp.html")

  cluster_and_render('sigmod_mixturemodel_dm', 'stats.db',
          algo_fn=mixturemodel_dirichlet_multinomial(pickle_file='states_mixturemodel_dm.p'),
          merge_fn=scipy_fcluster_merge_strategy(zmat_file_prefix='zmat_mixturemodel_dm'),
          outname="./text_mixturemodel_dm.html")
