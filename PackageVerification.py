import tensorflow as tf

import tensorflow_datasets as tfds

import pandas as pd

import numpy as np

import scipy

import PIL

if tf.__version__ == '2.7.0':
    print("TensorFlow version is correct", tf.__version__)
else:
    print("TensorFlow version is wrong, expected 2.7.0 received",tf.__version__)

if tfds.__version__ == '4.4.0':
    print("TensorFlow Dataset version is correct", tfds.__version__)
else:
    print("TensorFlow Dataset version is wrong, expected 4.4.0 received", tfds.__version__)

if PIL.__version__ == '8.4.0':
    print("Pillow version is correct", PIL.__version__)
else:
    print("Pillow version is wrong, expected 8.4.0 received", PIL.__version__)

if pd.__version__ == '1.3.4':
    print("Pandas version is correct", pd.__version__)
else:
    print("Pandas version is wrong, expected 1.3.4 received", pd.__version__)

if np.__version__ == '1.21.4':
    print("Numpy version is correct", np.__version__)
else:
    print("Numpy version is wrong, expected 1.21.4 received", pd.__version__)

if scipy.__version__ == '1.7.3':
    print("Sci Py version is correct", scipy.__version__)
else:
    print("Sci Py version is wrong, expected 1.21.4 received", scipy.__version__)
