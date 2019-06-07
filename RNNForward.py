from RNNForwardStep import RNNForwardStep
import numpy as np

def RNNForward(x, model):
    # Inputs:
    # - x: input data of size (N, T, D) per batch (see trainRNN, testRNN)
    # - model: RNN model containing weights and states.

    # Outputs:
    # - output: landmark prediction of size (N, M)  per batch
    # - model: RNN model containing weights
    # - state_h: updated states (useful for  backpropagation)

    # Weights/States from model you may want to use here
    # - state_h: hidden states for your batch - it has size (N, H, T)
    # - model.param['Wo']: weight matrix for hidden-to-output connections - it has size (H, M)
    # - model.param['bo']: biases - it has size (1, M)

    N, T, D = x.shape
    H, _ = model.param['Wh'].shape

    # h0: hidden state with all-zeros for first window frame - it has size of size (N, H)
    # h: hidden states for all window frames - it must have size (N, H, T)
    h0 = np.zeros(shape=(N, H))
    state_h = np.zeros(shape=(N, H, T))

    # stepwise forward propagation for the time-delayed RNN's hidden layer
    state_h[:,:,0] = RNNForwardStep(np.squeeze(x[:,0,:]), h0, model)
    for t in range(1, T):
        state_h[:,:,t] = RNNForwardStep(np.squeeze(x[:,t,:]), state_h[:,:,t-1], model)

    ''' ================================================================================ 
           YOUR CODE GOES HERE  
    ================================================================================ '''

    ## use state_h[:,:,-1] to compute the output!
    Wo = model.param['Wo']
    bo = model.param['bo']
    output = state_h[:,:,-1].dot(Wo) + bo

    return output, model, state_h
