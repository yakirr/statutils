from __future__ import print_function, division
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def qqplot(pvals, minuslog10p=False, text='', fontsize='medium', **kwargs):
    x = np.arange(1/len(pvals), 1+1/len(pvals), 1/len(pvals))[:len(pvals)]
    logx = -np.log10(x)
    maxy = 3*np.max(logx)
    if minuslog10p:
        logp = np.sort(pvals)[::-1]
    else:
        logp = -np.log10(np.sort(pvals))
    logp[logp >= maxy] = maxy
    l, r = min(np.min(logp), np.min(logx)), max(np.max(logx), np.max(logp))
    plt.scatter(logx, logp, **kwargs)
    plt.plot([l, r], [l, r], ls="--", c=".3")
    plt.xlim(min(logx), max(logx))
    plt.xlabel('-log10(rank/n)', fontsize=fontsize)
    plt.ylabel('-log10(p)', fontsize=fontsize)
    plt.title(text)

# creates a scatter plot where x is binned and y is averaged within each bin
def scatter_b(x, y, binsize=50, func=np.mean, **kwargs):
    # boundaries = np.linspace(min(x), max(x), nbins)
    boundaries = np.concatenate([np.sort(x)[::binsize],[np.max(x)]])
    bins =  zip(boundaries[:-1], boundaries[1:])
    print(len(bins), 'bins')

    binx, biny = np.empty(len(bins)), np.empty(len(bins))
    binx[:] = np.nan; biny[:] = np.nan
    for i, (l, r) in enumerate(bins):
        mask = (x>=l)&(x<r)
        # print(l,r, mask.sum())
        binx[i] = func(x[mask])
        biny[i] = func(y[mask])

    plt.scatter(binx, biny, **kwargs)
    return binx, biny

# creates a scatter plot where x is smoothed over some window
def scatter_s(x, y, windowsize=100, perwindow=10, **kwargs):
    import statutils.smooth as smooth
    df = pd.DataFrame({'x':x, 'y':y})
    df.sort(columns='x', inplace=True)
    x = df.x.values
    y = df.y.values
    xs = smooth.smooth(x, windowsize, stride=int(windowsize/perwindow))
    ys = smooth.smooth(y, windowsize, stride=int(windowsize/perwindow))
    plt.scatter(xs, ys)

# scatter plot with points colored by spatial density
def scatter_d(x, y, **kwargs):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=z, edgecolor='', **kwargs)

# creates a scatter plot with marginal distribution histograms. kwargs are passed to the
# scatter function
def scatter_m(x, y, xbins=None, ybins=None, text='', **kwargs):
    notnan = np.isfinite(x) & np.isfinite(y)
    if notnan.sum() < len(x):
        print('WARNING:', (~notnan).sum(), 'of', len(x), 'entries were nan or inf')
    x = x[notnan]; y = y[notnan]

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
