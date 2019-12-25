from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix

def cluster_accuracy(predicted: np.array , target: np.array):
    assert predicted.size == target.size, ''.join('Different size between predicted\
        and target, {} and {}').format(predicted.size, target.size)

    D = max(predicted.max(), target.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(predicted.size):
        w[predicted[i], target[i]] += 1
    
    ind_1, ind_2 = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_1, ind_2)]) * 1.0 / predicted.size, w

def plot_grad_flow(named_parameters):
    '''
    __author__: discuss.pytorch.org / RoshanRane
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def plot_grad_flow_lines(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def assign_cluster_2_reads(groups, y_grp_cl):
    label_cl_dict=dict()

    for idx, g in enumerate(groups):
        for r in g:
            label_cl_dict[r]=y_grp_cl[idx]
    
    y_cl=[]
    for i in sorted(label_cl_dict):
        y_cl.append(label_cl_dict[i])
    
    return y_cl

def eval_quality(y_true, y_pred, n_clusters=NUM_OF_SPECIES):
    A = confusion_matrix(y_pred, y_true)
    if len(A) == 1:
      return 1, 1
    prec = sum([max(A[:,j]) for j in range(0,n_clusters)])/sum([sum(A[i,:]) for i in range(0,n_clusters)])
    rcal = sum([max(A[i,:]) for i in range(0,n_clusters)])/sum([sum(A[i,:]) for i in range(0,n_clusters)])

    return prec, rcal