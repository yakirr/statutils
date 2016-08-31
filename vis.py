from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bin_on_x(x, y, nbins = 25):
    df = pd.dataframe({'x':x, 'y':y})
    df['group_index'] = map(int, (nbins-0.001)*(x-min(x))/(max(x)-min(x)))
    for g in range(nbins):
        pass
    #TODO

def qqplot(pvals, minuslog10p=False, fname=None, show_anyway=False, text=''):
    x = np.arange(1/len(pvals), 1+1/len(pvals), 1/len(pvals))[:len(pvals)]
    logx = -np.log10(x)
    if minuslog10p:
        logp = np.sort(pvals)[::-1]
    else:
        logp = -np.log10(np.sort(pvals))
    l, r = min(np.min(logp), np.min(logx)), max(np.max(logx), np.max(logp))
    plt.scatter(logx, logp)
    plt.plot([l, r], [l, r], ls="--", c=".3")
    plt.xlim(min(logx), max(logx))
    plt.xlabel('-log10(rank/n)')
    plt.ylabel('-log10(p)')
    plt.title(text)

    if fname:
        plt.savefig(fname)
        if show_anyway:
            plt.show()
        plt.clf()
    else:
        plt.show()

