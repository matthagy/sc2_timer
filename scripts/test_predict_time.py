
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

from autopath import datadir, modeldir
import slicetime; reload(slicetime)
from slicetime import slice_time, ColonMissingError

with closing(gzip.open(os.path.join(datadir, 'extract_timer.p.gz'))) as fp:
    data = pickle.load(fp)

with open(os.path.join(modeldir, 'rf_digits.p')) as fp:
    rf = pickle.load(fp)


def predict_number(el):
    a, = rf.predict(el[0].ravel()[np.newaxis, ::])
    b, = rf.predict(el[1].ravel()[np.newaxis, ::])
    if a == 11:
        a = 0
    assert b != 11
    return 10*a + b

plt.clf()
i = 0
for seq in data:
    print 'seq'
    acc = []
    for img in seq:
        if not i % 20:
            print i
        i += 1
        try:
            tm = np.array(slice_time(img))
        except ColonMissingError:
            print 'colon missing error'
            continue
        mn = predict_number(tm[0])
        s = predict_number(tm[1])
        acc.append(mn + s / 60.0)
    if acc:
        plt.plot(acc, 'k-')
plt.xlabel('Frame')
plt.ylabel('Time (minutes)')
plt.draw()
plt.show()
plt.savefig('frame_time.png', bbox_inches='tight')
