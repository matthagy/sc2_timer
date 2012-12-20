
from __future__ import division

import os
import os.path
from contextlib import closing

import numpy as np
import matplotlib.pyplot as plt
import Image

from autopath import datadir
from util import create_composite_grid_image

digits_path = os.path.join(datadir, 'digits.npy')
classification_path = os.path.join(datadir, 'digits_classifications.npy')


images = np.load(digits_path)
classifications = np.load(classification_path)

mask = classifications != -1
images = images[mask]
classifications = classifications[mask]

indicies = np.argsort(classifications)

comp = create_composite_grid_image(images[indicies])
comp = comp.resize(tuple(2 * np.array(comp.size)))
comp.save('examples.png')
