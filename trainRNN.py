import numpy as np
import matplotlib.pyplot as plt
import os
import RNNForward
import RNNBackward


class opt():
    def __init__(self):
        # initialize default options
        self.num_hidden_nodes_rnn_layer = 256
        self.time_step = 25
        self.iterations = 15
        self.learning_rate =  0.01
        self.momentum =  0.9
        self.weight_decay =  0.001



def trainRNN(X, Y, neutral_face, options=opt()):

    ''' ================================
        initialize mode
    ================================ '''
    D = X.shape[1] # number of input features (in this assignment: # audio features pre-extracted for you per frame)
    M = Y.shape[1] # number of outputs - for this assignment, 38 2D facial landmark (x, y) positions concatenated in a vector per frame
    H = options.num_hidden_nodes_rnn_layer # number of RNN hidden nodes

    class Model():
        def __init__(self):
            self.num_nodes = [D, H, M]
            self.num_layers = len(self.num_nodes)
            self.weight_decay = options.weight_decay

            print('Number of RNN layers (including input and output layer):', self.num_layers)
            print('Number of RNN nodes in each layer will be:', self.num_nodes)
            print('Will run for {} iterations'.format(options.iterations))
            print('Learning rate: {}, Momentum: {}'.format(options.learning_rate, options.momentum))

            # Initialize model parameters
            # - Wx: Weight matrix for input-to-hidden layer connections of size (D, H)
            # - Wh: Weight matrix for hidden-to-hidden layer connections of size (H, H)
            # - Wo: Weight matrix for hidden-to-output layer connections of size (H, M)
            # - b: Biases for hidden layer of size (1, H)
            # - bo: Biases for output of size (1, M)
            self.param = {}
            self.param['Wx'] = 2.0 / (D + H) * np.random.randn(D, H)
            self.param['Wh'] = 2.0 / (H + H) * np.random.randn(H, H)
            self.param['b'] = np.zeros((1, H))
            self.param['Wo'] = 2.0 / (M + H) * np.random.randn(H, M)
            self.param['bo'] = np.zeros((1, M))

            # you may want to use these variables to store current derivatives (for implementing momentum)
            self.deriv = {}
            for key in self.param:
                self.deriv[key] = np.zeros_like(self.param[key])

    model = Model()

    ''' ================================
        prepare input
    ================================ '''
    # reshape input by aggregating audio features within a small window of
    # past frames (of size 'options.time_step')
    # your toy RNN will take as input the features within each small window
    # produce hidden states per each window frame, and use the hidden state
    # of the last window frame to output a single prediction.
    # this kind of ''time-delayed'' or many-to-one RNN has the benefit
    # of considering a broader temporal context to perform better predictions.
    # There are  several other more advanced variations (LSTMs, GRUs, other RNN
    # variations:  http://karpathy.github.io/2015/05/21/rnn-effectiveness/ ,
    # which we don't need to consider for this assignment.
    # due to this aggregation, the input will now become [N, T, D] where:
    # N: number of training instances
    # T: size of window
    # D: number of input features

    X_train = np.zeros((X.shape[0] - options.time_step, options.time_step, D))
    for i in range(0, X.shape[0] - options.time_step):
        X_train[i,:,:] = X[i : i + options.time_step, : ]
    N = X_train.shape[0]  # total number of reshaped training frames

    # Y has size N x M
    Y_train = Y - np.tile(neutral_face, (X.shape[0], 1))  # subtract neutral landmark face, only consider the displacement
    Y_train = Y_train[options.time_step:]  # cutting the first 'options.time_step' frames due to the used time delay

    ''' ================================================================
        train the model through batch gradient descent
    ================================================================ '''
    # initialize iteration (epoch) index, batch size, and training loss placeholder for
    # making a plot of loss wrt each epoch
    batch_size = 100
    training_loss_for_plot = np.zeros((options.iterations))

    for iter in range(options.iterations):
        shuffle_idx = np.random.permutation(N) #shuffle the order of training data for each iteration
        for ib in range(int(N/batch_size)):
            # pick training data in batch size from shuffled index
            picked_batch_idx = shuffle_idx[ib * batch_size : (ib + 1) * batch_size]
            x = X_train[picked_batch_idx, :, :]
            y = Y_train[picked_batch_idx, :]

            ''' =================================================================================== 
                   YOUR CODE GOES HERE - do forward/backward propagation and update weights     
            =================================================================================== '''

            # TODO: forward propagatation
            # RNNForward(?);
            output, model, state_h = RNNForward.RNNForward(x, model)
            
            # Loss
            Loss = np.linalg.norm(y - output)**2

            # TODO:  back propagatation
            output_msg = 2*(y - output)  # dLdY
            # RNNBackward(?);
            model = RNNBackward.RNNBackward(output_msg, x, model, state_h)

            # TODO:  update weight matrix with momentum
            # model.param['Wx'] = ?
            # model.param['Wh'] = ?
            # model.param['b'] = ?
            # model.param['Wo'] = ?
            # model.param['bo'] = ?

            ''' ==========================================================
                               END OF YOUR CODE 
            ========================================================== '''


        ''' ================================================================================ 
               YOUR CODE GOES HERE - get the prediction of the landmakrs and traning loss     
        ================================================================================ '''

        # TODO: produce preduction for each training frame and calculate loss
        # Y_output = ?
        train_loss = 0  # change this line obviously

        ''' ==========================================================
                           END OF YOUR CODE 
        ========================================================== '''

        # visualize training progress (loss function wrt epoch index)
        training_loss_for_plot[iter] = train_loss
        plt.plot(training_loss_for_plot[0:iter+1], 'k.-')
        plt.pause(0.01)

        print('Iteration {}, Cost function: {}'.format(iter, train_loss))

    print('Pause here - close the figure to continue')
    plt.show()

    return model, options
