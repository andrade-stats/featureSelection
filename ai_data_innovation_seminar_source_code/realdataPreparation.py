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
import pandas as pd
from matplotlib import pyplot 

numpy.set_printoptions(precision=3, suppress=True)

# "diabetesWithNA.csv" corresponds to dataset "PimaIndiansDiabetes2" from 
# http://www.cit.ctu.edu.vn/~dtnghi/detai/PimaIndiansDiabetes.html
df = pd.read_csv("datasets/diabetesWithNA.csv")

X, y = df.drop("diabetes", axis=1), df["diabetes"]
variableNames = X.columns.to_numpy()

X = X.to_numpy()
y = y.to_numpy()

# replace undefined values (NAN) by mean value of each feature (attribute)
allColumnMeans = numpy.nanmean(X, axis=0)
allColumnsMeansAsMatrix = numpy.outer(numpy.ones(X.shape[0]), allColumnMeans)
X[numpy.isnan(X)] = allColumnsMeansAsMatrix[numpy.isnan(X)]

# convert response to 0/1
y[y == "pos"] = 1
y[y == "neg"] = 0
y = y.astype(numpy.int)

# save as numpy array
numpy.save("datasets/diabetes_X", X, allow_pickle = False)
numpy.save("datasets/diabetes_y", y, allow_pickle = False)
numpy.save("datasets/diabetes_explanatory_variable_names", variableNames, allow_pickle = True)

# # show correlations between features
# correlations = df.corr(method = 'pearson') 
# print(correlations)

# # # draw Box-Plot
# df.plot(kind= 'box', subplots=True, layout=(3,3), sharex=False, sharey=False)
# pyplot.show()


print("FINISHED")