def mean_of_last_k(l, k):
    import numpy as np
    l = l if len(l) <= k else l[-k:]
    return np.mean(l)
