from __future__ import print_function
from __future__ import division

import os
import glob
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from skimage.io import imread, imsave
from scipy.misc import imresize

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('subset', False, 'If true, build a subset.')
flags.DEFINE_integer('downsample', 20, 'Downsample factor.')

print (FLAGS.downsample)
