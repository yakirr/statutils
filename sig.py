from __future__ import print_function, division
import numpy as np
from scipy import stats

def fdr(pvals, minuslog10p=False, threshold=0.05):
    if minuslog10p:
        pvals = np.power(10, -pvals)
    pvals.sort()
    cs = np.arange(1,len(pvals)+1)*threshold/len(pvals)
    cutoff_p = 0
    for c,p in zip(cs, pvals):
        if p < c:
            cutoff_p = p
    return cutoff_p, np.sum(pvals <= cutoff_p)


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
