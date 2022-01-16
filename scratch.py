import numpy as np

you = list(map(float, "-0.9031807 -1.0374491 -1.468 -1.321 0.931".split()))
i = list(map(float, "-1.06 -0.91 -0.31 -0.57 1.04".split()))

from sklearn.metrics.pairwise import cosine_similarity

you = np.array(you).reshape(1, -1)
i = np.array(i).reshape(1, -1)
print(cosine_similarity(you, i))