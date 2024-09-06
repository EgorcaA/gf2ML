
import numpy as np
from triqs_maxent import *


def findExtrema(inputArray):
   arrayLength = len(inputArray)
   outputCount = 0
   for k in range(1, arrayLength - 1):
      outputCount += (inputArray[k] > inputArray[k - 1] and inputArray[k] > inputArray[k + 1])
      outputCount += (inputArray[k] < inputArray[k - 1] and inputArray[k] < inputArray[k + 1])
   return outputCount

def gaussian(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def get_Gw(G_tau):
   tm = TauMaxEnt(cost_function='bryan', probability='normal')
   tm.omega = HyperbolicOmegaMesh(omega_min=-1.5, omega_max=1.5, n_points=101)
   tm.set_G_tau(G_tau)
   err = 5.e-4
   tm.set_error(err)
   result = tm.run()
   A0 = result.analyzer_results['Chi2CurvatureAnalyzer']['A_out']
   A0ens = result.omega
   return A0ens, A0 
