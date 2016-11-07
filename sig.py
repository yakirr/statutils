from __future__ import print_function, division
import numpy as np
from scipy import stats


def st_fdr(pvals, threshold=0.05):
    qvals, pi0 = info(pvals)
    numsig = (qvals <= threshold).sum()
    return np.nan, numsig, pi0

def info(pvals):
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

def sigrows(df, pvals, threshold=0.05, qname='q'):
    qvals, _ = info(pvals)
    sigdf = df[qvals <= threshold]
    sortedindices = [x for (y,x) in sorted(zip(qvals[qvals<=threshold],sigdf.index.values))]
    return sigdf.loc[sortedindices]


# st = use Storey-Tibshirani R package
def fdr(pvals, minuslog10p=False, threshold=0.05, st=True):
    if minuslog10p:
        pvals = np.power(10, -pvals)
    pvals = pvals[~np.isnan(pvals)] # filter out nans so we don't count them as tests

    if st:
        return st_fdr(pvals, threshold=threshold)
    else:
        pvals = np.sort(pvals)
        cs = np.arange(1,len(pvals)+1)*threshold/len(pvals)
        cutoff_p = 0
        for c,p in zip(cs, pvals):
            if p < c:
                cutoff_p = p
        return cutoff_p, np.sum(pvals <= cutoff_p), np.nan


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
