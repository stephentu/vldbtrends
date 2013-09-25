import math
import sqlite3
import sklearn
import numpy as np
import pdb

from sklearn.cluster import KMeans
from sompy import SOM
from util import Canonicalizer
from collections import *
import random
import prettyplotlib as ppl

# This is "import matplotlib.pyplot as plt" from the prettyplotlib library
from prettyplotlib import plt

# This is "import matplotlib as mpl" from the prettyplotlib library
from prettyplotlib import mpl


db = sqlite3.connect('stats.db')

def vectors():
  r = db.execute("select word, year, c from counts order by word, year")
  vects = defaultdict(dict)
  for w,y,c in r:
    l = vects[w]
    l[y] = c

  ret = []
  for w in vects:
    d = vects[w]
    counts = [float(d.get(y, 0.)) for y in xrange(1975, 2014)]
    smooth = []
    for i in xrange(len(counts)):
      smooth.append(np.mean(counts[i:i+3]))
    ret.append([w] + smooth)
  return np.array(ret)
  

vects = vectors()
clusterer = KMeans(8, n_init=30, init='k-means++')
data = vects[:,1:].astype(float)
data = np.array([l / max(l) for l in data ])
clusterer.fit(data)
labels = clusterer.labels_
xs = np.array(range(1975, 2014))

imgs = []
content = []

def add_content(subcluster, content, suffix):
    fig, ax = plt.subplots(1, figsize=(13,5))
    subcluster = sorted(subcluster, key=lambda t: max(t[1:].astype(float)), reverse=True)[:10]
    subcluster = np.array(subcluster)
    words = subcluster[:,0]
    ys = subcluster[:,1:].astype(float)
    mean = [np.mean(ys[:,i]) for i in xrange(ys.shape[1])]
    ys = ys.transpose()
    ppl.plot(ax, xs, ys, alpha=0.25, color="#7777ee")
    ppl.plot(ax, xs, mean, alpha=1, color="black")
    fname = './plots/plot%s.png' % (suffix)
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
  if not len(cluster): continue
  cluster = np.array(cluster)
  words = cluster[:,0]
  words = list(words)

  if 'crowd' in words:
    data = cluster[:,1:].astype(float)
    data = np.array([l / max(l) for l in data])
    clusterer = KMeans(2)
    clusterer.fit(data)
    for newlabel in set(clusterer.labels_):
      idxs = clusterer.labels_ == newlabel
      subcluster = cluster[idxs]
      add_content(subcluster, content, "%s-%s" % (label, newlabel))
        
    continue


  cluster = cluster[:10]
  add_content(cluster, content, label)

content.sort(key=lambda c: c[-1])

from jinja2 import Template
template = Template(file('./clustertemplate.html').read())


with file("./test.html", 'w') as f:
  f.write( template.render(content=content))