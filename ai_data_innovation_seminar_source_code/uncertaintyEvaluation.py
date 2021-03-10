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
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.calibration
import dataGeneration

import matplotlib.pyplot as plt
import matplotlib.ticker
    
import dataPreparation

# checked
# y = true class labels {0,1}
# predictedTrueClassProbs = probabilities P(Y = 1) 
def getReliabilityDiagramData(y, predictedTrueClassProbs, nrBins, binningType = "EqualProbabilitySize"):
    assert(y.shape[0] == predictedTrueClassProbs.shape[0])
    
    assert(numpy.min(predictedTrueClassProbs) >= 0.0 and numpy.max(predictedTrueClassProbs) <= 1.0)
        
    if binningType == "EqualProbabilitySize":
        # create binning such that each bin covers the same length of probability
        binsWithBorders = numpy.linspace(start = 0.0, stop = 1.0, num = nrBins + 1)
        
    elif binningType == "EqualSampleSize":
        # create binning such that each bin contains the same number of samples
        quantileProbabilities = numpy.linspace(start = 0.0, stop = 1.0, num = nrBins+1)
        
        binsWithBorders = numpy.quantile(predictedTrueClassProbs, quantileProbabilities)
        binsWithBorders[0] = 0.0
        binsWithBorders[len(binsWithBorders)-1] = 1.0
        
        # print(binsWithBorders)
        # for i in range(nrBins):
        #    print(numpy.sum(numpy.logical_and(predictedTrueClassProbs <= binsWithBorders[i+1], predictedTrueClassProbs >= binsWithBorders[i])))
    else:
        assert(False)
        
           
    binsBorders = numpy.copy(binsWithBorders)
    binsWithBorders[len(binsWithBorders) - 1] += 0.00001 # to ensure that we cover 1.0 when we use "numpy.digitize"
    binIds = numpy.digitize(predictedTrueClassProbs, binsWithBorders) - 1 # subtract 1 because id's start at 1
    
    proportionTrueEachBin = numpy.zeros(nrBins)
    predictedTrueProbEachBin = numpy.zeros(nrBins)
    nrSamplesEachBin = numpy.zeros(nrBins, dtype = numpy.int)
    for i in range(nrBins):
        predictedTrueProbEachBin[i] = numpy.mean(predictedTrueClassProbs[binIds == i])
        proportionTrueEachBin[i] = numpy.mean(y[binIds == i])
        nrSamplesEachBin[i] = numpy.sum(binIds == i)
    
    return proportionTrueEachBin, predictedTrueProbEachBin, nrSamplesEachBin, binsBorders



# partly adapted from 
# https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py
def plotReliabilityDiagram(proportionTrueEachBin, predictedTrueProbEachBin, nrSamplesEachBin, binBorders, filename = ""):
    
    # filter out bins for which there is no data
    proportionTrueEachBin = proportionTrueEachBin[nrSamplesEachBin != 0]
    predictedTrueProbEachBin = predictedTrueProbEachBin[nrSamplesEachBin != 0]

    plt.figure(figsize=(3.5, 7.0))
    
    upperFig = plt.subplot2grid((2, 1), (0, 0)) 
    lowerFig = plt.subplot2grid((2, 1), (1, 0)) 
    
    new_tick_labels = [str(round(v,1)) for v in binBorders]
    
    import matplotlib.ticker as plticker
    loc = plticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals
    upperFig.yaxis.set_major_locator(loc)
    upperFig.set_xlabel("Predicted probability")
    upperFig.set_ylabel("Success rate")
    upperFig.set_ylim([-0.05, 1.05])
    upperFig.set_xticks(binBorders)
    upperFig.set_xticklabels(new_tick_labels)
    # upperFig.legend(loc="lower right")
    upperFig.set_title("Reliability Diagram")

    upperFig.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    upperFig.plot(predictedTrueProbEachBin, proportionTrueEachBin, "s-", label="%s (%1.3f)" % ("logistic regression", 0))
    

    lowerFig.set_ylabel("Number of samples")
    lowerFig.set_xlabel("Predicted probability")
    lowerFig.set_xlim(upperFig.get_xlim())      
    lowerFig.set_xticks(binBorders)
    lowerFig.set_xticklabels(new_tick_labels)
    lowerFig.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    
    binWidths = numpy.diff(binBorders)
    
    xMiddlePositions = binBorders[:-1] + binWidths/2
    
#     print("binBorders = ", binBorders)
#     print("binWidths = ", binWidths)
#     print("binBorders[:-1] = ", binBorders[:-1])
#     print("xMiddlePositions = ", xMiddlePositions)
#     print("nrSamplesEachBin = ", nrSamplesEachBin)
    # print("len(binBorders) == ", len(binBorders))
    # assert(False)
    # lowerFig.bar(xMiddlePositions, nrSamplesEachBin, width = 1.0 / (len(binBorders) - 1), edgecolor = "black")
    lowerFig.bar(xMiddlePositions, nrSamplesEachBin, width = binWidths, edgecolor = "black")

    plt.tight_layout()
    
    if len(filename) == 0:
        plt.show()
    else:
        plt.savefig(filename + ".png")
    
    return



def showReliabilityDiagram(y, predictedTrueClassProbs, nrBins):
    proportionTrueEachBin, predictedTrueProbEachBin, nrSamplesEachBin, binBorders = getReliabilityDiagramData(y, predictedTrueClassProbs, nrBins)
    
    # filter out bins for which there is no data
    proportionTrueEachBin = proportionTrueEachBin[nrSamplesEachBin != 0]
    predictedTrueProbEachBin = predictedTrueProbEachBin[nrSamplesEachBin != 0]
    
    plotReliabilityDiagram(proportionTrueEachBin, predictedTrueProbEachBin, nrSamplesEachBin, binBorders)
    return


# calculates expected calibration error (ECE)
def getECE(proportionTrueEachBin, predictedTrueProbEachBin, nrSamplesEachBin):
    dataRatioInEachBin = (nrSamplesEachBin / numpy.sum(nrSamplesEachBin))
    errorInEachBin = numpy.abs(proportionTrueEachBin - predictedTrueProbEachBin)
    weightedErrorInEachBin = numpy.multiply(dataRatioInEachBin, errorInEachBin)
    ECE = numpy.sum(weightedErrorInEachBin[nrSamplesEachBin > 0])
    return ECE


def getAverageNLL(allClassProbs, y):
    return - numpy.mean(numpy.log(allClassProbs[numpy.arange(y.shape[0]), y]))

def getAccuracy(allClassProbs, y):
    predictedLabels = numpy.argmax(allClassProbs, axis = 1)
    return (1.0 / y.shape[0]) * numpy.sum(predictedLabels == y)


def test():

    print("x = ")
    y_test = numpy.asarray([0,0,1,1,1,1,1,1])
    predictedTrueClassProbs = numpy.asarray([0.0, 0.1, 0.21, 0.24, 0.79, 0.9, 0.99, 1.0])
    
    # y_test = numpy.asarray([1,0,1,1,0,1,1,1])
    # predictedTrueClassProbs = numpy.asarray([0.79, 0.1, 0.99, 0.24, 0.0, 0.9, 0.21, 1.0])
    
    # perfect calibration example
    # proportionTrueEachBin = numpy.linspace(start = 0.1, stop = 0.9, num = 9)
    # predictedTrueProbEachBin = numpy.copy(proportionTrueEachBin)
    # nrSamplesEachBin = numpy.ones(10) * 10
    # binBorders = numpy.linspace(start = 0.0, stop = 1.0, num = 11)
    # print(proportionTrueEachBin)
    # print("binBorders = ", binBorders)
    # plotCalibrationCurve(proportionTrueEachBin, predictedTrueProbEachBin, nrSamplesEachBin, binBorders)
