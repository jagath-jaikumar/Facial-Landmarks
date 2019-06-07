import numpy as np
from CreateAvi import  CreateAvi


def testRNN(data, model, options):
    # apply trained RNN model on 3 different test sequences
    for test_idx in range(3):

        # reshape test input into similar shape as training input - it will have size [N, T, D]
        X_test = data['X_test'][0, test_idx]
        x = np.zeros(shape=(X_test.shape[0]-options.time_step, options.time_step, X_test.shape[1]))
        for i in range(0, X_test.shape[0]-options.time_step):
            x[i,:,:] = X_test[i: i + options.time_step,:]

        Y_test = data['Y_test'][0, test_idx]
        y = Y_test - np.tile(data['neutral_face'], (X_test.shape[0], 1)) # subtract neutral landmark face, only consider the displacement
        y = y[options.time_step:] # cutting the first 'options.time_step' frames due to the used time delay (our prediction will have a small latency!)

        ''' ================================================================================ 
               YOUR CODE GOES HERE - get the prediction of the landmakrs and traning loss     
        ================================================================================ '''

        # TODO: produce output predictions and test loss
        Y_output = np.zeros_like(y) # change this line obviously
        test_loss = 0  # change this line obviously

        ''' ==========================================================
                           END OF YOUR CODE 
        ========================================================== '''

        print('Test set {}, Average L2 loss: {}'.format(test_idx, test_loss))

        # A post-processing for temporal smoothing is useful given that we used a simple, shallow RNN.
        # We also visualize the prediction results here.
        Y_output += np.tile(data['neutral_face'], (Y_output.shape[0], 1)) # prediction
        y += np.tile(data['neutral_face'], (Y_output.shape[0], 1)) # ground-truth

        def smooth(x, N):
            return np.convolve(x, np.ones((N,)) / N)[(N - 1):]
        for i in range(Y_output.shape[1]):
            Y_output[:, i] = smooth(Y_output[:, i], 15)
            y[:, i] = smooth(y[:, i], 15)

        # draw ground-truth in red on left, and your estimation in blue on right
        # saved as pred_xxx.avi, where xxx = test_idx

        # ONLY CREATE VIDEO FOR THE FIRST TEST SEQUENCE, i.e. test_0.wav
        # CreateAvi(Y_output, y, 'pred_{}.avi'.format(test_idx))
