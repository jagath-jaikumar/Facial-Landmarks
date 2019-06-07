import time
import numpy as np
from scipy.io import loadmat
from showFacialLandmarks import showFacialLandmarks
from trainRNN import trainRNN
from testRNN import testRNN
import matplotlib.pyplot as plt

# run script

# load train/test dataset
data = loadmat('starter_data.mat') # command this line if data is loaded

for key in data.keys():
    try:
        print(key, data[key].shape)
    except:
        pass


# want to see the training facial landmarks? Uncomment the following lines

# for i in range(0, data['Y_train'].shape[0]):
#     plt = showFacialLandmarks(data['Y_train'][i, :], 0, 'r-')
#     plt.pause(0.01)
#     plt.clf()
#
# plt.show()

# Uncomment the following line to see a single, neutral (silent) face.
# The net will produce displacements of landmarks relative to this neutral face

# plt = showFacialLandmarks(data['neutral_face'][0], 1.5, 'b-')
# plt.show()

# train the RNN model
# the code will run on your CPU (it might take around 10-30 min to finish training depending on your machine)
# obviously, it would be more desirable to train a deep RNN on GPUs, yet
# for the purpose of this course assignment, a small RNN is enough
#
start_time = time.time()
model, options  = trainRNN(data['X_train'], data['Y_train'], data['neutral_face'])
print('Time usage:', time.time() - start_time)
#
# # test model on test sequences (will use trained model & options)
testRNN(data, model, options)


