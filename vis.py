from __future__ import print_function, division
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy.stats import gaussian_kde, beta
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt


#############
# qq plotting

def qqplot(pvals, minuslog10p=False, text='', fontsize='medium', errorbars=True,
        ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    x = np.arange(1/len(pvals), 1+1/len(pvals), 1/len(pvals))[:len(pvals)]
    logx = -np.log10(x)
    maxy = 3*np.max(logx)
    if minuslog10p:
        logp = np.sort(pvals)[::-1]
    else:
        logp = -np.log10(np.sort(pvals))
    logp[logp >= maxy] = maxy
    l, r = min(np.min(logp), np.min(logx)), max(np.max(logx), np.max(logp))

    if errorbars:
        ranks = np.arange(1,len(logp)+1)
        cilower = -np.log10(beta.ppf(.025, ranks, len(logx)-ranks +1))
        ciupper = -np.log10(beta.ppf(.975, ranks, len(logx)-ranks +1))
        ax.fill_between(logx, cilower, ciupper,
                facecolor='gray', interpolate=True, alpha=0.2,
                linewidth=0)

    ax.scatter(logx, logp, **kwargs)
    ax.plot([l,r], [l,r],
            c='gray', linewidth=0.2, dashes=[1,1])
    ax.set_xlim(min(logx), 1.01*max(logx))
    ax.set_xlabel(r'$-\log_{10}(\mathrm{rank}/n)$', fontsize=fontsize)
    ax.set_ylabel(r'$-\log_{10}(p)$', fontsize=fontsize)
    ax.set_title(text)
    plt.tight_layout()

#############
# scatter plots

# creates a scatter plot where x is smoothed over some window
def scatter_s(x, y, windowsize=100, perwindow=10, ax=None, **kwargs):
    import statutils.smooth as smooth
    if ax is None:
        ax = plt.gca()
    df = pd.DataFrame({'x':x, 'y':y})
    df = df[x.notnull() & y.notnull()]
    df.sort_values('x', inplace=True)
    x = df.x.values
    y = df.y.values
    xs = smooth.smooth(x, windowsize, stride=int(windowsize/perwindow))
    ys = smooth.smooth(y, windowsize, stride=int(windowsize/perwindow))
    ax.scatter(xs, ys, **kwargs)
    ax.set_xlim(1.05*min(xs), 1.05*max(xs))
    ax.set_ylim(1.05*min(ys), 1.05*max(ys))
    return xs, ys

# creates a scatter plot where x is binned and y is averaged within each bin
# if extreme_only is an integer then all bins will be lumped together other than the most
#   exteme extreme_only bins on either side
def scatter_b(x, y, binsize=50, func=np.mean,
        extreme_only=None, left_only=None, right_only=None,
        errorbars=False,
        ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    # boundaries = np.linspace(min(x), max(x), nbins)
    boundaries = np.concatenate([np.sort(x)[::binsize],[np.max(x)]])
    if extreme_only is not None:
        boundaries = np.concatenate([boundaries[:extreme_only], boundaries[-extreme_only:]])
    if left_only is not None:
        boundaries = np.concatenate([boundaries[:left_only], boundaries[-1:]])
    if right_only is not None:
        boundaries = np.concatenate([boundaries[:1], boundaries[-right_only:]])
    bins =  zip(boundaries[:-1], boundaries[1:])
    print(len(bins), 'bins')

    binx, biny, std = np.empty(len(bins)), np.empty(len(bins)), np.empty(len(bins))
    binx[:] = np.nan; biny[:] = np.nan
    for i, (l, r) in enumerate(bins):
        mask = (x>=l)&(x<r)
        # print(l,r, mask.sum())
        binx[i] = func(x[mask])
        biny[i] = func(y[mask])
        std[i] = np.std(y[mask]) / np.sqrt(mask.sum())

    if not errorbars:
        std=None
    ax.errorbar(binx, biny, yerr=std, **kwargs)
    ax.set_xlim(1.3*min(binx), 1.3*max(binx))
    ax.set_ylim(1.3*min(biny), 1.3*max(biny))
    return binx, biny

# scatter plot with points colored by spatial density
def scatter_d(x, y, **kwargs):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=z, edgecolor='', **kwargs)

# hex bin plot
# TODO

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

#############
# displaying matrices

# cluster and display correlation matrix
# note: column names must be unique
def cluster_and_show(corrdf, names=None, thresh=0):
    import seaborn as sns

    if names is not None:
        corrdf = corrdf.rename(columns={c:n for c,n in zip(corrdf.columns, names)})
    corrdf.index = corrdf.columns

    # distance = ssd.squareform(1-np.abs(corrdf.values))
    # Y = sch.linkage(distance)
    Y = sch.linkage(corrdf.values, method='centroid') # if the above lines aren't working

    Z = sch.dendrogram(Y, orientation='right', no_plot=True)

    ind = Z['leaves']
    displaydf = corrdf.iloc[ind][corrdf.columns[ind]]
    sns.heatmap(displaydf.applymap(lambda x: 0 if np.abs(x) < thresh else x),
            xticklabels=True,
            yticklabels=True,
            square=True,
            vmin=-1,
            vmax=1)
    plt.xticks(rotation=90, fontsize=8)

    return displaydf
    # plt.xticks(np.arange(len(names)), names[ind], rotation='vertical', fontsize=8)
    # plt.yticks(np.arange(len(names)), names[ind], fontsize=8)
    # plt.colorbar()
    # plt.gcf().tight_layout()
