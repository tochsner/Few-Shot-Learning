import scipy as sp
from scipy import stats
import numpy as np
from data.MNIST import *
from helper.prepareTriplets import *

W = np.array([[0,0,1],[1,0,2],[2,4,5],[4,7,2]])
v = np.array([0,1,2,3])[np.newaxis].T
v_2 = np.array([1,-1,0])[np.newaxis].T

data = load_data()
data = prepare_data_for_keras(data)
g_data = group_data(data[0])

createTrainingDataForQuadrupletLoss(None, g_data, 100, 20)          