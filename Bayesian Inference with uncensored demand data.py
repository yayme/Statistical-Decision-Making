import numpy as np
import scipy.stats
import scipy.optimize
from matplotlib import pyplot

def log_likelihood_censored( theta, Xs, qs ):
  ans=0.0
  for i in range(len(Xs)):
    if Xs[i]<qs[i]:
      ans+=scipy.stats.geom.logpmf(Xs[i],theta)
    else:
      ans+=np.log(1-scipy.stats.geom.cdf(qs[i]-1,theta))
  return ans

def posterior_density_censored( theta, Xs, qs ):
  # IMPLEMENT YOUR CODE HERE
  log_likelihood=log_likelihood_censored(theta,Xs,qs)
  log_prior=scipy.stats.beta.logpdf(theta,a=1,b=1)

  return np.exp(log_likelihood+log_prior)
def MAP_censored( Xs, qs ):
  # IMPLEMENT YOUR CODE HERE
  objective=lambda x: -posterior_density_censored(x,Xs,qs)
  result = scipy.optimize.minimize_scalar(objective, bounds=(0.001, 0.999), method='bounded')
  return result.x if result.success else None
def posterior_mean_censored( Xs, qs ):
  # IMPLEMENT YOUR CODE HERE
  int_num=lambda theta: posterior_density_censored(theta,Xs,qs)*theta
  int_den=lambda theta: posterior_density_censored(theta,Xs,qs)
  num=scipy.integrate.quad(int_num,0,1)[0]
  den=scipy.integrate.quad(int_den,0,1)[0]
  return num/den if den!=0 else None
Ds = np.array([16, 25, 18, 16, 11, 21, 12, 44, 65, 10])
qs = np.array([20, 20, 20, 20, 20, 30, 30, 30, 30, 30])
Xs = np.minimum( Ds, qs )

theta_MAP_censored = MAP_censored(Xs, qs)
theta_PM_censored = posterior_mean_censored(Xs, qs)

thetas = np.linspace(0.001,0.1,100)
pyplot.plot( thetas, [posterior_density_censored(t, Xs, qs) for t in thetas] )
pyplot.axvline( theta_MAP_censored, c='r', ls='--' )
pyplot.axvline( theta_PM_censored, c='g', ls='--' )

pyplot.grid(True)
pyplot.xlabel( r"$\theta$" )
pyplot.ylabel( r"Posterior density" )
pyplot.show()
print( "MAP result = %f" % theta_MAP_censored )
print( "posterior mean = %f" % theta_PM_censored )