
from __future__ import division

import os
import os.path
from contextlib import closing

import numpy as np
import matplotlib.pyplot as plt
import Image

from autopath import datadir

digits_path = os.path.join(datadir, 'digits.npy')
classification_path = os.path.join(datadir, 'digits_classifications.npy')


images = np.load(digits_path)
classifications = np.load(classification_path)

mask = classifications != -1
images = images[mask]
classifications = classifications[mask]

acc = []
factor = 3
plt.figure(1)
plt.clf()
for i,d in enumerate(range(10) + [11]):
    plt.subplot(3, 4, i+1)
    c = images[classifications == d]
    c = np.array([np.asarray(Image.fromarray(i).convert('L'))
                  for i in c])
    plt.imshow(c.mean(axis=0), interpolation='nearest', vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.draw()
plt.show()
plt.savefig('avg_digits.png', bbox_inches='tight')
