import numpy as np
import scipy.stats
from matplotlib import pyplot
import scipy.optimize

def log_likelihood_censored( theta, Xs, qs ):
  ans=0.0
  for i in range(len(Xs)):
    if Xs[i]<qs[i]:
      ans+=scipy.stats.geom.logpmf(Xs[i],theta)
    else:
      ans+=np.log(1-scipy.stats.geom.cdf(qs[i]-1,theta))
  return ans
def MLE_censored( Xs, qs ):
  objective=lambda x: -log_likelihood_censored(x,Xs,qs)
  result = scipy.optimize.minimize_scalar(objective, bounds=(0.001, 0.999), method='bounded')
  return result.x if result.success else None

Ds = np.array([16, 25, 18, 16, 11, 21, 12, 44, 65, 10])
qs = np.array([20, 20, 20, 20, 20, 30, 30, 30, 30, 30])
Xs = np.minimum( Ds, qs )

theta_MLE_censored = MLE_censored(Xs, qs)

thetas = np.linspace(0.001,0.1,100)
pyplot.plot( thetas, [log_likelihood_censored(t, Xs, qs) for t in thetas] )
pyplot.plot( theta_MLE_censored, log_likelihood_censored(theta_MLE_censored, Xs, qs), '*', ms=10 )

pyplot.grid(True)
pyplot.xlabel( r"$\theta$" )
pyplot.ylabel( r"log-likelihood" )
pyplot.show()

print( "MLE result = %f" % theta_MLE_censored )