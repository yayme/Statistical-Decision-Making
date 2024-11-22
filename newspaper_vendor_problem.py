import numpy as np
import scipy.stats
from matplotlib import pyplot
import scipy.optimize
def log_likelihood_uncensored( theta, Ds ):

  ans=0.0
  for i in range(len(Ds)):
    ans+=scipy.stats.geom.logpmf(Ds[i],theta)
  return ans

def MLE_uncensored( Ds ):

  # return scipy.optimize.minimize_scalar(lambda x: -log_likelihood_uncensored(x, Ds)).x
  return len(Ds)/np.sum(Ds)

Ds = np.array([16, 25, 18, 16, 11, 21, 12, 44, 65, 10])

theta_MLE_uncensored = MLE_uncensored(Ds)

thetas = np.linspace(0.001,0.1,100)
pyplot.plot( thetas, [log_likelihood_uncensored(t, Ds) for t in thetas] )
pyplot.plot( theta_MLE_uncensored, log_likelihood_uncensored(theta_MLE_uncensored, Ds), '*', ms=10 )

pyplot.grid(True)
pyplot.xlabel( r"$\theta$" )
pyplot.ylabel( r"log-likelihood" )


print( "MLE result = %f" % theta_MLE_uncensored )
pyplot.show()

