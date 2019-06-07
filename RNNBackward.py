import numpy as np
from RNNBackwardStep import  RNNBackwardStep

def RNNBackward(output_msg, x, model, state_h):
    # Inputs:
    # - output_msg: back-propage message from top output layer - size (N, M).
    # - x: Input data for the entire timeseries - size (N, T, D).
    # - model: RNN model containing weights, states, derivatives
    # - state_h: updated states (calculated from  forward passing)

    # Outputs:
    #   - model: RNN model containing weights and updated derivatives

    # Weights/States from model you may want to use
    # - state_h: hidden states for the current batch - size(N, H, T)  (computed during forward propagation)
    # - model.param['Wx']: weight matrix for input-to-hidden connections - size (D, H)
    # - model.param['Wh']: weight matrix for hidden-to-hidden connections - size (H, H)
    # - model.param['Wo']: weight matrix for hidden-to-output connections - size (H, M)
    # - model.param['b']: biases for hidden state, of shape (1, H)
    # - model.param['bo']: biases for output, of shape (1, M)

    N, H, T = state_h.shape
    N, T, D = x.shape
    N, M = output_msg.shape

    # derivatives for each weight matrix in model.param.
    model.param['dWx'] = np.zeros((D, H))
    model.param['dWh'] = np.zeros((H, H))
    model.param['dWo'] = np.zeros((H, M))
    model.param['db'] = np.zeros((1, H))
    model.param['dbo'] = np.zeros((1, M))

    ''' ================================================================================ 
           YOUR CODE GOES HERE  
    ================================================================================ '''
    dL = output_msg
    ## output layer backward propagation
    dWo = state_h[:,:,-1].T.dot(dL)
    dbo = np.sum(dL, axis=0)
    dxo = dL.dot(dWo.T)
    model.param['dWo'] = dWo
    model.param['dbo'] = dbo
    dnext_h = dxo

    ## rnn step backward propagation
    # use RNNBackwardStep function for each time step within the window
    # i.e., you need to have a backwards loop: for t = T:-1:1   ...
    # and use backward_fc_layer_msg
    
    for t in range(T-1, -1, -1):
        dnext_h, dWx, dWh, db = RNNBackwardStep(dnext_h, x[:,t,:], model, t, state_h)

        model.param['dWx'] += dWx
        model.param['dWh'] += dWh
        model.param['db'] += db

    ## L2 regularization (weight_decay)
    Wo = model.param['Wo']
    bo = model.param['bo']
    Wx = model.param['Wx']
    Wh = model.param['Wh']
    b = model.param['b']
    
    reg = model.weight_decay
    model.param['dWo'] += 2 * reg * Wo
    model.param['dWx'] += 2 * reg * Wx
    model.param['dWh'] += 2 * reg * Wh

    return model
