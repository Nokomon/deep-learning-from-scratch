import sys
sys.path.append('..')

import numpy as np

# from ch04.negative_sampling_layer import *
#
# corpus = np.array([0,1,2,3,4,1,2,3])
# power = 0.75
# sample_size = 2
#
# sampler = UnigramSampler(corpus, power, sample_size)
# target = np.array([1,3,0])
# negative_sample = sampler.get_negative_sample(target)
# print(negative_sample)

a = np.arange(20).reshape(4,5)
print(a)
print(a[:,2])
print(a[2, :])