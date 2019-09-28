from scipy.optimize import linear_sum_assignment
import numpy as np

def cluster_accuracy(predicted: np.array , target: np.array):
    assert predicted.size == target.size, ''.join('Different size between predicted\
        and target, {} and {}').format(predicted.size, target.size)

    D = max(predicted.max(), target.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(predicted.size):
        w[predicted[i], target[i]] += 1
    
    ind_1, ind_2 = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_1, ind_2)]) * 1.0 / predicted.size, w