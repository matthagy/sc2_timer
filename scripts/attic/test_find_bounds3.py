
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
import slicetime; reload(slicetime)
from slicetime import slice_time

with closing(gzip.open(os.path.join(datadir, 'extract_timer.p.gz'))) as fp:
    data = pickle.load(fp)

plt.clf()

imgs = [img for seq in data for img in seq]
random.shuffle(imgs)

plt.figure(1)
plt.clf()
r = c = 1
a,b = slice_time(imgs[0])
for i,d in enumerate(a + b):
    plt.subplot(1, 4, i+1)
    print d.shape
    plt.imshow(d, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
plt.draw()
plt.show()
