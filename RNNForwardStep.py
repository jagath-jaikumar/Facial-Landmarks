import numpy as np

def RNNForwardStep(x, prev_h, model):
    # Inputs:
    # - x: Input data for this timestep -  size (N, D).
    # - prev_h: hidden state from previous timestep - size (N, H)
    # - model: trained model containing weights.

    # Outputs a tuple of:
    # - next_h: Next hidden state - size (N, H)

    # Weights/States from model you may want to use
    # - model.param['Wx']: weight matrix for input-to-hidden connections - size (D, H)
    # - model.param['Wh']: weight matrix for hidden-to-hidden connections  - size (H, H)
    # - model.param['b']: biases of shape (1, H)

    N, D = x.shape

    ''' ================================================================================ 
           YOUR CODE GOES HERE - compute hidden state     
    ================================================================================ '''
    Wx = model.param['Wx']
    Wh = model.param['Wh']
    b = model.param['b']
    
    hhh = prev_h.dot(Wh)
    hih = x.dot(Wx)
    
    next_h = np.tanh(hhh + hih + b)

    return next_h