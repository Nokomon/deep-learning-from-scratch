import sys
sys.append('..')

import numpy as np
from common import config
config.GPU = True

import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.util import *
from dataset import ptb

