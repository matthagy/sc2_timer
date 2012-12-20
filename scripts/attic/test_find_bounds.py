
from __future__ import division

import os
import os.path
import gzip
import cPickle as pickle
from contextlib import closing

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
for i,img in enumerate(imgs):
    if not i%250:
        print i
    delta = np.abs(img - band[::, np.newaxis, ::]).sum(axis=2).sum(axis=0)
    mins.append(delta.min())
    loc_mins.append(np.argmin(delta))

plt.figure(1)
plt.clf()
plt.hist(mins, 50)
plt.draw()

plt.figure(2)
plt.clf()
plt.hist(loc_mins, 50)
plt.draw()

plt.figure(3)
plt.clf()
plt.plot(mins, loc_mins, 'ko', alpha=0.1)
plt.draw()

plt.show()
