from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy import stats

def simes(pvals):
    pvals = np.sort(pvals)
    return np.min(pvals / np.arange(1, len(pvals)+1) * len(pvals))

# gets q-values from the storey-tibshirani R package
def st_info(pvals):
    from rpy2.robjects import r, pandas2ri
    pandas2ri.activate()
    from rpy2.robjects.packages import importr
    q = importr('qvalue')
    def py(x):
        return pandas2ri.ri2py(x)

    qobj = q.qvalue(pvals)
    pi0 = py(qobj.rx2('pi0'))[0]
    qvals = py(qobj.rx2('qvalues'))
    return qvals, pi0

# st = use Storey-Tibshirani R package
def fdr(pvals, minuslog10p=False, threshold=0.05, st=True):
    if minuslog10p:
        pvals = np.power(10, -pvals)
    pvals = pvals[~np.isnan(pvals)] # filter out nans so we don't count them as tests

    if st:
        qvals, pi0 = st_info(pvals)
        numsig = (qvals <= threshold).sum()
        return np.nan, numsig, pi0
    else:
        pvals = np.sort(pvals)
        cs = np.arange(1,len(pvals)+1)*threshold/len(pvals)
        less = (pvals <= cs)
        if less.sum() > 0:
            cutoff_p = np.max(pvals[less])
        else:
            cutoff_p = -1
        return cutoff_p, np.sum(pvals <= cutoff_p), np.nan

# return significant rows of a dataframe
# st can be True, False, or 'auto'
def sigrows(df, pvals, threshold=0.05, st=True):
    if st=='auto':
        st = (len(df) >= 40)
    if st:
        qvals, _ = st_info(pvals)
        sigdf = df[qvals<=threshold]
        sigsort = qvals[qvals<=threshold]
    else:
        thresh, _, _ = fdr(pvals, threshold=threshold, st=st)
        sigdf = df[pvals<=thresh]
        sigsort = pvals[pvals<=thresh]
    sortedindices = [x for (y,x) in sorted(zip(sigsort,range(len(sigdf))))]
    return sigdf.iloc[sortedindices]

# st is unused
def sigrows_onesidedz(df, zs, threshold=0.05, st=None):
    zts = np.arange(1,5,0.05)
    neg = np.array([(zs <= -t).sum() for t in zts], dtype=np.float)
    pos = np.array([(zs >= t).sum() for t in zts], dtype=np.float)
    rate = neg/(pos+neg)
    if any(rate <= threshold):
        best = min(zts[rate<=0.05])
    else:
        best = np.inf
    sigdf = df[zs>=best]
    sigsort = -zs[zs>=best]
    sortedindices = [x for (y,x) in sorted(zip(sigsort,range(len(sigdf))))]
    return sigdf.iloc[sortedindices]

def sigrows_strat(df, sig_measure, strat, threshold=0.05, st=False, exclude=None,
        sigrowsfunc=sigrows, output=False):
    if exclude is not None:
        df = df[~exclude]
        sig_measure = sig_measure[~exclude]
        strat = strat[~exclude]
    result = pd.DataFrame()
    num_tested = 0
    for l in np.unique(strat):
        some = df[strat == l]
        if len(some) == 0:
            continue
        else:
            num_tested += 1
        mysigrows = sigrowsfunc(some, sig_measure[strat==l], threshold=threshold, st=st)
        if output:
            print(l, len(some), len(mysigrows))
        result = pd.concat([result, mysigrows], axis=0)
    print('tested', num_tested, 'tranches, expected', threshold*num_tested,
        'results under null. Found', len(result), 'results total')
    return result



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--zscores')
    parser.add_argument('--threshold', type = float, default = 0.05)
    parser.add_argument('--pvals')
    args = parser.parse_args()

    if args.zscores is not None:
        z = np.array(map(float, open(args.zscores).readlines()))
        p = 2*stats.norm.sf(np.abs(z))
    else:
        p = np.array(map(float, open(args.pvals).readlines()))
    cutoff_p, numsig = fdr(p, threshold=args.threshold)
    print(cutoff_p)
    print(numsig, 'passed cutoff')
