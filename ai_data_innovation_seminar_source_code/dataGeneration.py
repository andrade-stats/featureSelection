# Copyright (C) 2021  Daniel Andrade
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; version 2
# of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy
import scipy.stats
import sklearn.datasets

def generateLogRegData(beta, correlationMatrix, intercept, n):
    
    meanVec = numpy.zeros(beta.shape[0])
    X = numpy.random.multivariate_normal(meanVec, correlationMatrix, n)
    logOdds = numpy.matmul(X, beta) + intercept
    
    occurrenceProbabilities = 1.0 / (1.0 + numpy.exp(-logOdds))
     
    # print("occurrenceProbabilities = ")
    # with numpy.printoptions(precision=3):
    #    print(occurrenceProbabilities)
    
    y = scipy.stats.bernoulli.rvs(occurrenceProbabilities)
    
    return X, y



def generateCorrelationMatrix(d, rho):

    correlationMatrix = numpy.zeros((d,d))
    for i in range(d):
        for j in range(d):
            correlationMatrix[i,j] = rho ** numpy.abs(i-j)
        
    return correlationMatrix
    
    
# generate data as in 
# "Contraction properties of shrinkage priors in logistic regression", Journal of Statistical Planning and Inference, 2020
# increase intrinisic (aleotory uncertainty) by lowering "scale"
def generateExampleData1(n, d, scale = 1.0, rho = 0.2):
    
    intercept = 0.0
     
    prespecifiedBetaPart = numpy.asarray([1, 1.5, -2, 2.5]) * scale
    beta = numpy.zeros(d)
    beta[0:prespecifiedBetaPart.shape[0]] = prespecifiedBetaPart
        
    
    correlationMatrix = generateCorrelationMatrix(d, rho)
    
    nonZeroCoefficientsVec = numpy.zeros(d, dtype = numpy.int) 
    nonZeroCoefficientsVec[0:prespecifiedBetaPart.shape[0]] = 1

    return generateLogRegData(beta, correlationMatrix, intercept, n), nonZeroCoefficientsVec



def generateNonLinearData(n, d):
    assert(d >= 2)
    
    X_2dPart, y = sklearn.datasets.make_gaussian_quantiles(cov=1.0,
                                     n_samples=n, n_features=2,
                                     n_classes=2, random_state=44231)
    
    
    correlationMatrix = generateCorrelationMatrix(d-2, rho = 1.0)
    meanVec = numpy.zeros(d-2)
    X_remaining = numpy.random.multivariate_normal(meanVec, correlationMatrix, n)
    
    X = numpy.hstack((X_2dPart, X_remaining))
    
    return X, y


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    X, y = generateNonLinearData(n = 100, d= 8)

    colormap = numpy.array(['b', 'r'])
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:,0], X[:,1], marker='o', c=colormap[y], s=25)

    plt.tight_layout()
    plt.show()
    