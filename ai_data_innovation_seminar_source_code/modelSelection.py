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
import sklearn.linear_model
import dataGeneration

import dataPreparation
import uncertaintyEvaluation
import sklearn.metrics

import scipy.special
import cv
import const

def setResult(allNrSelectedVariables, allF1_scores, allHeldOutNLL, runId, selectedModelPerformance, allScores):
    modelId = numpy.argmin(allScores)
    selectedModelPerformance[runId, const.NR_VARIABLES_ID] = allNrSelectedVariables[modelId]
    selectedModelPerformance[runId, const.F1_SCORE_ID] = allF1_scores[modelId]
    selectedModelPerformance[runId, const.HOLD_OUT_LOGLIKELIHOOD_ID] = - allHeldOutNLL[modelId]
    return selectedModelPerformance

numpy.random.seed(3523421)


NR_RUNS = 50

selectedModelPerformance_BIC = numpy.zeros(shape = (NR_RUNS, 3))
selectedModelPerformance_AIC = numpy.zeros(shape = (NR_RUNS, 3))
selectedModelPerformance_CV = numpy.zeros(shape = (NR_RUNS, 3))


realData = True

if not realData:
    n = 200 # size of training data
    d = 20 # number of covariates

    NR_ORACLE_TEST_SAMPLES = 10000
    (X_original, y_original), groundTruthNonZeros = dataGeneration.generateExampleData1(n + NR_ORACLE_TEST_SAMPLES, d, rho = 0.1)
else:
    NR_ORACLE_TEST_SAMPLES = 100
    X_original = numpy.load("datasets/diabetes_X.npy", allow_pickle = False)
    y_original = numpy.load("datasets/diabetes_y.npy", allow_pickle = False)
    variableNames = numpy.load("datasets/diabetes_explanatory_variable_names.npy", allow_pickle = True)
    assert(X_original.shape[1] == variableNames.shape[0])


for runId in range(NR_RUNS):
    X, y, X_oracleTest, y_oracleTest = dataPreparation.splitTrainingAndTest(X_original, y_original, NR_ORACLE_TEST_SAMPLES)

    print("d = ", X.shape[1])
    print("n = ", X.shape[0])
    assert(False)

    allRegParamValues = numpy.logspace(start = -3, stop = 3, num=100)

    foundModels = set()

    for regParamValue in allRegParamValues:
        print("regParamValue = ", regParamValue)
        model = sklearn.linear_model.LogisticRegression(penalty = "l1",  solver = "liblinear", fit_intercept = True, C = regParamValue)
        model.fit(X, y)
        betaVector = numpy.reshape(model.coef_, (-1)) # convert to vector
        selectedCovariateIds = numpy.nonzero(betaVector)[0] # get indices of nonzero entries in beta
        print("selectedCovariateIds = ", selectedCovariateIds)
        foundModels.add(selectedCovariateIds.tostring()) # convert to string to make it hashable

        
    allModels = []
    allScores_AIC = []
    allScores_BIC = []
    allScores_CV  = []

    allHeldOutNLL = []
    allAccuracies = []
    allBrierScores = []

    allF1_scores = []
    allNrSelectedVariables = []

    for selectedCovariateIdsAsStr in foundModels:
        selectedCovariateIds = numpy.fromstring(selectedCovariateIdsAsStr, dtype=numpy.int)
        
        # find Maximum-Likelihood(ML) estimate by setting (almost) no l2-penalty (high value of C)
        modelForML = sklearn.linear_model.LogisticRegression(penalty = "l2",  fit_intercept = True, C = 1000.0)
        
        # print("sklearn.metrics.SCORERS.keys() = ", sklearn.metrics.SCORERS.keys())
        # assert(False)
        # cv_results = sklearn.model_selection.cross_validate(modelForML, X[:, selectedCovariateIds], y, cv=3) # , scoring=("neg_log_loss"), return_train_score=False)
        
        # selectedX = X[:, selectedCovariateIds]
        # print(selectedX.shape)
        # assert(False)
        # cv_results = sklearn.model_selection.cross_val_score(modelForML, X[:, selectedCovariateIds], y, cv=3) # , scoring=("neg_log_loss"), return_train_score=False)
        # print("scores = ") 
        #print(scores)
        # print(cv_results)
        # print(numpy.asarray(cv_results))
        # print(cv_results['test_score'])
        # assert(False)

        

        if selectedCovariateIds.shape[0] > 0:
            # use scipy library to calculate ML estimate
            modelForML.fit(X[:, selectedCovariateIds], y)
            allClassProbs = modelForML.predict_proba(X[:, selectedCovariateIds])
            negativeLogLikelihood = y.shape[0] * uncertaintyEvaluation.getAverageNLL(allClassProbs, y)
            
            allClassProbsOracleTest = modelForML.predict_proba(X_oracleTest[:, selectedCovariateIds])
            
        else:
            # beta is constant zero vector, calculate ML estimate on my own 
            classOneRatio =  numpy.mean(y)
            tau = scipy.special.logit(classOneRatio) # ML estimate of tau
            nrOneLabels = numpy.sum(y)
            nrZeroLabels = y.shape[0] - nrOneLabels
            negativeLogLikelihood = - nrOneLabels * numpy.log(scipy.special.expit(tau)) - nrZeroLabels * numpy.log(scipy.special.expit(-tau))
            # this is just equivalent to 
            # negativeLogLikelihood = - nrOneLabels * numpy.log(classOneRatio) - nrZeroLabels * numpy.log(1.0 - classOneRatio)
            
            allClassProbsOracleTest =  numpy.tile([1.0 - classOneRatio, classOneRatio], (X_oracleTest.shape[0], 1)) 
        
        
        nrFreeParameters = selectedCovariateIds.shape[0] + 1
        AIC = 2.0 * negativeLogLikelihood + 2.0 * nrFreeParameters
        allScores_AIC.append(AIC)
        BIC = 2.0 * negativeLogLikelihood + nrFreeParameters * numpy.log(y.shape[0])
        allScores_BIC.append(BIC)
        allModels.append(selectedCovariateIds)

        allScores_CV.append(cv.runCrossvalidationForBinaryClassification(modelForML, X, y, selectedCovariateIds))

        if not realData:
            selectedCovariatesVec = numpy.zeros_like(groundTruthNonZeros)
            selectedCovariatesVec[selectedCovariateIds] = 1
            f1 = sklearn.metrics.f1_score(groundTruthNonZeros, selectedCovariatesVec)
        else:
            f1 = 0.0
        allF1_scores.append(f1)
        
        allNrSelectedVariables.append(selectedCovariateIds.shape[0])
        
        # calculate performance on hold-out data
        avgNegLogLikelihood =  uncertaintyEvaluation.getAverageNLL(allClassProbsOracleTest, y_oracleTest)
        acc = uncertaintyEvaluation.getAccuracy(allClassProbsOracleTest, y_oracleTest)
        brierScore = sklearn.metrics.brier_score_loss(y_oracleTest, allClassProbsOracleTest[:,1])
        allHeldOutNLL.append(avgNegLogLikelihood)
        allAccuracies.append(acc)
        allBrierScores.append(brierScore)

    setResult(allNrSelectedVariables, allF1_scores, allHeldOutNLL, runId, selectedModelPerformance_BIC, allScores_BIC)
    setResult(allNrSelectedVariables, allF1_scores, allHeldOutNLL, runId, selectedModelPerformance_AIC, allScores_AIC)
    setResult(allNrSelectedVariables, allF1_scores, allHeldOutNLL, runId, selectedModelPerformance_CV, allScores_CV)

    # print("model ranking:")
    # for modelRank, modelId in enumerate(sortedModelIds):
    #     print("**********")
    #     print("Top " + str(modelRank + 1) + ": " + str(allModels[modelId]))
    #     print("BIC = " + str(allScores_BIC[modelId]))
    #     print("AIC = " + str(allScores_AIC[modelId]))
    #     print("CV = " + str(allScores_CV[modelId]))
    #     print("nr selected variables = ", str(allNrSelectedVariables[modelId]))
    #     if realData:
    #         print(variableNames[allModels[modelId]])
    #     else:
    #         print("f1 score = ", allF1_scores[modelId])
    #     print("NLL = " + str(allHeldOutNLL[modelId]))
    #     print("Brier Score = " + str(allBrierScores[modelId]))
    #     print("Accuracy = " + str(allAccuracies[modelId]))



print("selectedModelPerformance_BIC = ")
print(selectedModelPerformance_BIC)
print("FINISHED ALL EXPERIMENTS")

if not realData:
    dataPrefix = "simulatedData"
else:
    dataPrefix = "realData"

numpy.save("results/" + dataPrefix + "_performance_BIC", selectedModelPerformance_BIC)
numpy.save("results/" + dataPrefix + "_performance_AIC", selectedModelPerformance_AIC)
numpy.save("results/" + dataPrefix + "_performance_CV", selectedModelPerformance_CV)


