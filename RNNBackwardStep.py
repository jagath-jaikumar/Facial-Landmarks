import numpy as np

def RNNBackwardStep (dnext_h, x, model, t, state_h):
    # Inputs:
    #  - dnext_h: backpropagated message from hidden state of next timestep/output layer - size (N, H)
    #  - x: Input data for batch - size (N, T, D).
    #  - model: trained model containing weights, states, derivatives
    #  - t: time step index
    #  - state_h: updated states (calculated from  forward passing)

    # Outputs:
    #  - dprev_h: backpropagated message towards previous timestep  - size (N, H)
    #  - dWx: Derivative of weight matrix for input-to-hidden connections, Wx
    #  - dWh: Derivative of weight matrix for hidden-to-hidden connections, Wh
    #  - db: Derivative of biases for hidden state, b

    # Weights/States from model you may want to use
    #  - state_h: hidden states for the entire timeseries - size (N, H, T)
    #  - model.param['Wx']: weight matrix for input-to-hidden connections - size (D, H)
    #  - model.param['Wh']: weight matrix for hidden-to-hidden connections - size (H, H)
    #  - model.param['Wo']: weight matrix for hidden-to-output connections - size (H, M)
    #  - model.param['b']: biases for hidden state - size (1, H)
    #  - model.param['bo']: biases for output - size  (1, M)

    x = np.squeeze(x[:,t,:])  # squeeze the time-step dimension

    ''' ================================================================================ 
           YOUR CODE GOES HERE - get the derivatives and message to send out    
    ================================================================================ '''
    Wx = model.param['Wx']
    Wh = model.param['Wh']
    Wo = model.param['Wo']
    b = model.param['b']
    bo = model.param['bo']
    
    dhihphhh = dnext_h * (1 - state_h[t]**2)
    
    dWx = x.T.dot(dhihphhh)
    dx = dhihphhh.dot(Wx.T)
    
    dWh = state_h[t-1].T.dot(dhihphhh)
    dprev_h = dhihphhh.dot(Wh.T)
    
    db = np.sum(dhihphhh, axis=0)

    return dprev_h, dWx, dWh, db
