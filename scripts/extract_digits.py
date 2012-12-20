
from __future__ import division

import os
import os.path
import gzip
import cPickle as pickle
from contextlib import closing
import random

import numpy as np

from autopath import datadir
import slicetime; reload(slicetime)
from slicetime import slice_time, ColonMissingError

with closing(gzip.open(os.path.join(datadir, 'extract_timer.p.gz'))) as fp:
    data = pickle.load(fp)

i = 0
acc = {}
for seq in data:
    acc_seq = []
    for img in seq:
        i += 1
        if not i%500:
            print i
        try:
            tm = np.array(slice_time(img))
        except ColonMissingError:
            print 'colon missing error'
            continue
        for el in tm:
            for d in el:
                acc[d.tostring()] = d
digits = np.array(acc.values())
np.save(os.path.join(datadir, 'digits.npy'), digits)
