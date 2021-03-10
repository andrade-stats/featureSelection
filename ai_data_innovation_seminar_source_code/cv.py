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

import sklearn
import numpy
import scipy
import uncertaintyEvaluation

def runCrossvalidationForBinaryClassification(model, X_original, y_original, selectedCovariateIds):

    skf = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True)

    negativeLogLikelihood = 0.0

    for train_index, test_index in skf.split(X_original, y_original):
        X_train, X_test = X_original[train_index], X_original[test_index]
        y_train, y_test = y_original[train_index], y_original[test_index]

        if selectedCovariateIds.shape[0] == 0:
            classOneRatio =  numpy.mean(y_train)
            tau = scipy.special.logit(classOneRatio) # ML estimate of tau
            nrOneLabels = numpy.sum(y_test)
            nrZeroLabels = y_test.shape[0] - nrOneLabels
            negativeLogLikelihood += - nrOneLabels * numpy.log(scipy.special.expit(tau)) - nrZeroLabels * numpy.log(scipy.special.expit(-tau))
        else:
            model.fit(X_train[:, selectedCovariateIds], y_train)
            allClassProbs_predicted = model.predict_proba(X_test[:, selectedCovariateIds])
            negativeLogLikelihood += y_test.shape[0] * uncertaintyEvaluation.getAverageNLL(allClassProbs_predicted, y_test)
    
    return negativeLogLikelihood
