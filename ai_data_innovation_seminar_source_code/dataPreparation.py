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
import sklearn.preprocessing

def splitTrainingAndTest(X, y, testDataSize):
    
    rndIdOrder = numpy.arange(y.shape[0])
    numpy.random.shuffle(rndIdOrder)
    testDataIds = rndIdOrder[0:testDataSize]
    trainDataIds = rndIdOrder[testDataSize:y.shape[0]] 

    X_train = X[trainDataIds, :]
    y_train = y[trainDataIds]
    X_test = X[testDataIds, :]
    y_test = y[testDataIds]
        
    # dataScalerX = sklearn.preprocessing.StandardScaler().fit(X_train)
    dataScalerX = sklearn.preprocessing.RobustScaler().fit(X_train)
    X_train = dataScalerX.transform(X_train)
    X_test = dataScalerX.transform(X_test)
      
    return X_train, y_train, X_test, y_test 


def prepareForTensorFlow(X, y):
    # convert to float (to make tensorflow happy)
    X = X.astype(numpy.float32)
    
    # convert to one-hot encoding (to make tensorflow happy, since tensorflow does not support advanced indexing)
    yOneHot = numpy.zeros((y.shape[0], numpy.max(y) + 1), dtype = numpy.float32)
    yOneHot[numpy.arange(y.shape[0]), y] = 1
    
    return X, yOneHot


def getValidationData(X, y, nrValidationSamples):
    assert(X.shape[0] >= 2 * nrValidationSamples)
    n = X.shape[0]
    X_train = X[nrValidationSamples:n,:]
    y_train = y[nrValidationSamples:n]
    X_val = X[0:nrValidationSamples,:]
    y_val = y[0:nrValidationSamples]
    return X_train, y_train, X_val, y_val

