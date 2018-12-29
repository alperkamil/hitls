import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.special import logsumexp

def compress(a,wsize,type='mean'):
    N = a.shape[0]
    if type=='mean':
        return a[:(N//wsize)*wsize].reshape((N//wsize,wsize,)+a.shape[1:]).mean(axis=1)
    else:
        return np.median(a[:(N//wsize)*wsize].reshape((N//wsize,wsize,)+a.shape[1:]),axis=1)
    
def moving_average(a, n=5):
    ret = np.cumsum(a, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n:] = ret[n:] / n
    for i in range(n):
        ret[i] =ret[i]/(i+1)
    return ret

def get_pcomp(a,t=0.99999):
    pca = PCA()
    pca.fit(a)
    total_var = 0
    pcomp = 0
    while total_var < t:
        total_var += pca.explained_variance_ratio_[pcomp]
        pcomp += 1
    return pcomp

def get_score_diff(fwdlattice,smoothing=5):
    ll=logsumexp(fwdlattice,axis=1)
    lld = ll[1:]-ll[:-1]
    return np.append([0]*smoothing,moving_average(lld,n=smoothing))
    
def get_intervals(a):
    intervals = []
    for t in range(1,len(a)):
        if a[t]>a[t-1]:
            start = t
        elif a[t]<a[t-1]:
            intervals.append((start,t))
    return intervals

def heatmap(data, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    # row_labels, col_labels,
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    #ax.set_xticklabels(col_labels)
    #ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar