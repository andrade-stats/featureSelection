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
import const
import matplotlib.pyplot as plt

#️ realData = False
realData = True

typeId = const.NR_VARIABLES_ID
# typeId = const.F1_SCORE_ID
# typeId = const.HOLD_OUT_LOGLIKELIHOOD_ID


if not realData:
    dataPrefix = "simulatedData"
else:
    dataPrefix = "realData"

selectedModelPerformance_BIC = numpy.load("results/" + dataPrefix + "_performance_BIC.npy")
selectedModelPerformance_AIC = numpy.load("results/" + dataPrefix + "_performance_AIC.npy")
selectedModelPerformance_CV = numpy.load("results/" + dataPrefix + "_performance_CV.npy")



DATA = [selectedModelPerformance_CV[:, typeId], selectedModelPerformance_AIC[:, typeId], selectedModelPerformance_BIC[:, typeId]]
TICKLABELS = ["CV", "AIC", "BIC"]

fig, ax = plt.subplots()
ax.boxplot(DATA)
ax.set_xticklabels(TICKLABELS, fontsize = 20)
# ax.set_yticklabels(fontsize = 20)
ax.yaxis.set_tick_params(labelsize=16)

if typeId == const.NR_VARIABLES_ID and not realData:
    ax.set_ylim([0.0, 14.0])
elif typeId == const.NR_VARIABLES_ID and realData:
    ax.set_ylim([0.0, 9.0])


if typeId == const.NR_VARIABLES_ID:
    typePostfix = "nrVariables"
elif typeId == const.F1_SCORE_ID:
    typePostfix = "f1Score"
elif typeId == const.HOLD_OUT_LOGLIKELIHOOD_ID:
    typePostfix = "logLikelihood"
else:
    assert(False)

# plt.show()
plt.savefig("plots/resultPlot_" + dataPrefix + "_" + typePostfix + ".pdf")
print("FINISHED")