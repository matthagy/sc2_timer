
from __future__ import division

import os
import os.path
import gzip
import cPickle as pickle
from contextlib import closing
import random

import numpy as np
import Image
import matplotlib.pyplot as plt

from autopath import datadir

with closing(gzip.open(os.path.join(datadir, 'extract_timer.p.gz'))) as fp:
    data = pickle.load(fp)

band = np.array([
    [  0,   0,   0],
       [  1,   1,   1],
       [  0,   0,   0],
       [  0,   0,   0],
       [  0,   0,   0],
       [136, 224, 167],
       [ 47,  77,  57],
       [  0,   0,   0],
       [  0,   0,   0],
       [ 47,  77,  57],
       [136, 224, 167],
       [  0,   0,   0],
       [  0,   0,   0]], dtype=np.uint8)

plt.clf()

mins = []
loc_mins = []

imgs = [img for seq in data for img in seq]
random.shuffle(imgs)

plt.figure(1)
plt.clf()
r = c = 1
imgs = imgs[:r*c:]
for i,img in enumerate(imgs):
    delta = np.abs(img - band[::, np.newaxis, ::]).sum(axis=2).sum(axis=0)
    l = np.argmin(delta)
    m = delta[l]

    plt.subplot(r, c, i+1)
    plt.imshow(img, interpolation='nearest')

    plt.axvline(l, color='r')
    for off in [-1,1]:
        for i in xrange(0,2):
            for a in [3,11]:
                plt.axvline(l + off * (i*11 + a), color='r')

    #plt.xticks([])
    #plt.yticks([])

plt.draw()
plt.show()
