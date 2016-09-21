from __future__ import print_function, division
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt


def qqplot(pvals, minuslog10p=False, text=''):
    x = np.arange(1/len(pvals), 1+1/len(pvals), 1/len(pvals))[:len(pvals)]
    logx = -np.log10(x)
    maxy = 3*np.max(logx)
    if minuslog10p:
        logp = np.sort(pvals)[::-1]
    else:
        logp = -np.log10(np.sort(pvals))
    logp[logp >= maxy] = maxy
    l, r = min(np.min(logp), np.min(logx)), max(np.max(logx), np.max(logp))
    plt.scatter(logx, logp)
    plt.plot([l, r], [l, r], ls="--", c=".3")
    plt.xlim(min(logx), max(logx))
    plt.xlabel('-log10(rank/n)')
    plt.ylabel('-log10(p)')
    plt.title(text)

# creates a scatter plot where x is binned and y is averaged within each bin
def scatter_b(x, y, nbins=25, **kwargs):
    boundaries = np.linspace(min(x), max(x), nbins)
    bins =  zip(boundaries[:-1], boundaries[1:])

    binx, biny = np.empty(len(bins)), np.empty(len(bins))
    binx[:] = np.nan; biny[:] = np.nan
    for i, (l, r) in enumerate(bins):
        binx[i] = np.mean(x[(x>=l)&(x<r)])
        biny[i] = np.mean(y[(x>=l)&(x<r)])

    plt.scatter(binx, biny, **kwargs)

# creates a scatter plot where x is smoothed over some window
def scatter_s(x, y, windowsize=100, **kwargs):
    df = pd.DataFrame({'x':x, 'y':y})
    df.sort(columns='x', inplace=True)
    #TODO

# creates a scatter plot with marginal distribution histograms. kwargs are passed to the
# scatter function
def scatter_m(x, y, xbins=None, ybins=None, text='', **kwargs):
    if not xbins:
        xbins = int(min(len(x)/20, 100))
    if not ybins:
        ybins = int(min(len(y)/20, 100))

    gs = gridspec.GridSpec(2,2, width_ratios=[3,1], height_ratios=[1,3])
    ax10 = plt.subplot(gs[1,0])
    ax10.scatter(x,y, **kwargs); ax10.set_title(text)
    ax00 = plt.subplot(gs[0,0])
    ax00.hist(x, bins=xbins, normed=True)
    ax11 = plt.subplot(gs[1,1])
    ax11.hist(y, bins=ybins, orientation='horizontal', normed=True)
    plt.title(text)
