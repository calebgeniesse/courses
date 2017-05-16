from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N = x.shape[0]
    
    # reshape input into rows
    x_row = x.reshape(N, -1)

    # forward pass
    out = x_row.dot(w)  + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x.shape[0]
    
    # reshape input into rows
    x_row = x.reshape(N, -1)

    # backward pass
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)

    dw = x_row.T.dot(dout)
    dw = dw.reshape(w.shape)

    db = dout.sum(axis=0)
    db = db.reshape(b.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout.copy()
    dx[x <= 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        # compute mean, var using minibatch stats
        mu = x.mean(axis = 0)
        var = x.var(axis = 0)
        
        # normalize data
        x_mu = x - mu
        sqrt_var = np.sqrt(var + eps)
        inv_var = 1.0 / sqrt_var
        x_hat = x_mu * inv_var
        
        # scale, shift normalized data using gamma, beta
        gamma_x = gamma * x_hat
        out = gamma_x + beta 

        # store out, cache
        cache = (x, gamma, beta, mu, var, x_mu, sqrt_var, inv_var, x_hat, eps)
                  
        # update running mean and runing var
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # store out, cache
        out = x.copy()
        cache = (running_mean, running_var)

        # normalize data
        out -= running_mean
        out *= 1.0 / np.sqrt(running_var + eps)
        
        # scale, shift normalized data using gamma, beta
        out *= gamma
        out += beta 
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    # get values from cache
    (x, gamma, beta, mu, var, x_mu, sqrt_var, inv_var, x_hat, eps) = cache
    
    # (9) backprop through beta shifting
    dbeta = dout.sum(axis=0)
    dgamma_x = dout.copy()

    # (8) backprop through gamma scaling
    dgamma = np.sum(dgamma_x * x_hat, axis=0)
    dx_hat = dgamma_x * gamma

    # (7-4) backprop through variance intermediates
    dinv_var = np.sum(dx_hat * x_mu, axis=0)
    dx_mul = dx_hat * inv_var
    dsqrt_var = -1.0 / (sqrt_var**2) * dinv_var
    dvar = 0.5 * 1.0 / np.sqrt(var + eps) * dsqrt_var
    dsq = 1.0 / dout.shape[0] * np.ones(dout.shape) * dvar
    
    # (3-1) backprop through mean intermediates
    dx_mu2 = 2 * x_mu * dsq
    dx1 = (dx_mul + dx_mu2)
    dmu = - 1.0 * np.sum(dx_mul + dx_mu2, axis=0)
    dx2 = 1.0 / dout.shape[0] * np.ones(dout.shape) * dmu

    # (0) backprop through dx
    dx = dx1 + dx2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # get values from cache
    (x, gamma, beta, mu, var, x_mu, sqrt_var, inv_var, x_hat, eps) = cache
    N = x.shape[0]
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(x_mu * (var + eps) ** (-.5) * dout, axis=0)
    dx = gamma * inv_var * (N * dout - dbeta - dgamma * x_mu * inv_var) / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # probability that a neuron remains active (not dropped)
        p_active = 1 - p
        # sample dropout mask
        mask = (np.random.rand(*x.shape) < p_active) / p_active
        # apply mask
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        mask = None
        out = x.copy()
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # backprop: out = x * mask
        dx = mask * dout
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # conv params
    stride = conv_param.get('stride')
    pad = conv_param.get('pad')
    
    # input dims
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    # output dims (N, F, H_out, W_out)
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    out = np.zeros((N, F, H_out, W_out))

    # add zero-padding to each image in x
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')

    # iter images
    for n in range(N):
        # iter filters
        for f in range(F):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    # array slices
                    hslice = slice(h_out * stride, h_out * stride + HH)
                    wslice = slice(w_out * stride, w_out * stride + WW)
                    # array inds
                    inds_out   = (n, f, h_out, w_out)
                    inds_x_pad = (n, slice(x_pad.shape[1]), hslice, wslice)
                    # perform matrix multiplication
                    out[inds_out] = np.sum(x_pad[inds_x_pad] * w[f, :]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # extract cache
    x, w, b, conv_param = cache

    # conv params
    stride = conv_param.get('stride')
    pad = conv_param.get('pad')

    # input dims
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, H_out, W_out = dout.shape
                
    # init grads
    dx = np.zeros((N, C, H, W))
    dw = np.zeros((F, C, HH, WW))
    db = np.zeros((F))

    # add zero-padding to each image in x
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')

    # backprop to dw, db
    for f in range(F):

        # back prop to db
        db[f] = np.sum(dout[:, f, :, :])

        # backprop to dw
        for c in range(C):
            for ihh in range(HH):               
                for jww in range(WW):
                    # slices
                    hslice = slice(ihh, ihh + H_out * stride, stride)
                    wslice = slice(jww, jww + W_out * stride, stride)
                    # inds
                    inds_dw = (f, c, ihh, jww)
                    inds_dout = (slice(N), f, slice(H_out), slice(W_out))
                    inds_x_pad = (slice(N), c, hslice, wslice)
                    # perform mat mul
                    dw[inds_dw] = np.sum(dout[inds_dout] * x_pad[inds_x_pad])

    # backprop to dx
    for n in range(N):
        for ih in range(H):
            for jw in range(W):
                for f in range(F):
                    for h_out in range(H_out):
                        # imask
                        imask = np.zeros_like(w[f, :, :, :])
                        di = ih + pad - h_out * stride
                        if float(di < HH and di >= 0):
                            imask[:, di, :] = 1.0

                        for w_out in range(W_out):
                            # jmask
                            jmask = np.zeros_like(w[f, :, :, :])
                            dj = jw + pad - w_out * stride
                            if float(dj < WW and dj >= 0):
                                jmask[:, :, dj] = 1.0
                                
                            # wmask
                            wmask = w[f, :, :, :] * imask * jmask
                            wmask = wmask.sum(axis=(1, 2))

                            # inds
                            inds_dx = (n, slice(dx.shape[1]), ih, jw)
                            inds_dout = (n, f, h_out, w_out)

                            # backprop dx at n,i,j
                            dx[inds_dx] += dout[inds_dout] * wmask
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    # pool params
    H_pool = pool_param.get('pool_height')
    W_pool = pool_param.get('pool_width')
    stride = pool_param.get('stride')

    # input dims
    N, C, H, W = x.shape
    
    # output dims
    H_out = (H - H_pool) // stride + 1
    W_out = (W - W_pool) // stride + 1

    out = np.zeros((N, C, H_out, W_out))

    # iter images
    for n in range(N):
        # iter channels
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    hslice = slice(h_out*stride, h_out*stride + H_pool)
                    wslice = slice(w_out*stride, w_out*stride + W_pool)
                    out[n, c, h_out, w_out] = x[n, c, hslice, wslice].max()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    # extract cache
    x, pool_param = cache

    # pool params
    H_pool = pool_param.get('pool_height')
    W_pool = pool_param.get('pool_width')
    stride = pool_param.get('stride')

    # output dims
    N, C, H, W = x.shape

    dx = np.zeros((N, C, H, W))
    
    # input dims
    H_out = (H - H_pool) // stride + 1
    W_out = (W - W_pool) // stride + 1

    # iter images
    for n in range(N):
        # iter channels
        for c in range(C):
            for h_out in range(H_out):
                hslice = slice(h_out*stride, h_out*stride + H_pool)
                for w_out in range(W_out):
                    wslice = slice(w_out*stride, w_out*stride + W_pool)  
                    # backprop into dx using mask for xpool
                    dpool = x[n, c, hslice, wslice]
                    dmask = dpool == dpool.max()
                    dx[n, c, hslice, wslice] += dout[n, c, h_out, w_out] * dmask
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape

    # transpose to (N, H, W, C), then reshape to (NxHxW, C)
    x = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    # apply vanilla batchnorm forward pass on reshaped x
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    # reshape to (N, H, W, C), then transpose to (N, C, H, W)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    
    # transpose to (N, H, W, C), then reshape to (NxHxW, C)
    dout = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    # apply vanilla batchnorm backpass on reshaped dout
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)

    # reshape to (N, H, W, C), then transpose to (N, C, H, W)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
