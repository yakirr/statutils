from scipy import stats
import numpy as np

def qnorm(x):
    #if there are ties, they are broken by argsort.
    ranks = (np.argsort(np.argsort(x))+1.) / (len(x)+1)
    z = stats.norm.ppf(ranks)
    return z

def test_qnorm():
    import matplotlib.pyplot as plt
    x = np.random.choice(50, 1000)
    z = qnorm(x)
    plt.scatter(x, z)
    plt.show()
    plt.hist(z, bins=20)
    plt.show()

if __name__ == '__main__':
    test_qnorm()
