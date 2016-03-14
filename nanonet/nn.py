import numpy as np

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return np.reciprocal(1.0 + np.exp(-x))

def linear(x):
    return x

def softplus(x):
    return np.log1p(np.exp(x))

def relu(x):
    return np.where(x > 0.0, x, 0.0)

"""  Convention: inMat row major (C ordering) as (time, state)
"""
tang_nn_type = np.float64

class layer:
    """  Basic feedforward layer  
         out = f( inMat W + b )

    :param W: Weight matrix of dimension (|input|, size)
    :param b: Bias vector of length  size.  Optional with default of no bias.
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, W, b=None, fun=tanh):
        assert b is None or len(b) == W.shape[1]
        self.b = np.zeros(W.shape[1], dtype=tang_nn_type) if b is None else b
        self.W = W
        self.f = fun

    def in_size(self):
        return self.W.shape[0]

    def out_size(self):
        return self.W.shape[1]

    def run(self, inMat):
        assert self.in_size() == inMat.shape[1]
        return self.f(inMat.dot(self.W) + self.b)

class softmax:
    """  Softmax layer
         tmp = exp( inmat W + b )
         out = row_normalise( tmp ) 
 
    :param W: Weight matrix of dimension (|input|, size)
    :param b: Bias vector of length size.  Optional with default of no bias.
    """
    def __init__(self, W, b=None):
        assert b is None or len(b) == W.shape[1]
        self.b = np.zeros(W.shape[1], dtype=tang_nn_type) if b is None else b
        self.W = W

    def in_size(self):
        return self.W.shape[0]

    def out_size(self):
        return self.W.shape[1]

    def run(self, inMat):
        assert self.in_size() == inMat.shape[1]
        tmp =  inMat.dot(self.W) + self.b
        m = np.amax(tmp, axis=1).reshape((-1,1))
        tmp = np.exp(tmp - m)
        x = np.sum(tmp, axis=1)
        tmp /= x.reshape((-1,1))
        return tmp

class rnn_layer:
    """ A simple recurrent layer
        Step:  state_new = fun( [state_old, input_new] W + b )
               output_new = state_new

    :param W: Weight matrix of dimension (|input| + size, size)
    :param b: Bias vector of length  size.  Optional with default of no bias.
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, W, fun=tanh):
        assert W.shape[0] > W.shape[1]
        self.W = W
        self.fun = fun
        self.size = W.shape[0] - W.shape[1]

    def in_size(self):
        return self.size

    def out_size(self):
        return self.W.shape[1]

    def run(self, inMat):
        assert self.in_size() == inMat.shape[1]
        out = np.zeros((inMat.shape[0], self.out_size()), dtype=tang_nn_type)
        state = np.zeros(self.size, dtype=tang_nn_type)
        for i, v in enumerate(inMat):
            state = self.fun(np.concatenate((state, v)).dot(self.W))
            out[i] = state

class lstm_layer:
    def __init__(self, iW, lW, b=None, p=None):
        """ LSTM layer with peepholes.  Implementation is to be consistent with
        Currennt and may differ from other descriptions of LSTM networks (e.g.
        http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

        Step:
            v = [ input_new, output_old ]
            Pforget = sigmoid( v W2 + b2 + state * p1)
            Pupdate = sigmoid( v W1 + b1 + state * p0)
            Update  = tanh( v W0 + b0 )
            state_new = state_old * Pforget + Update * Pupdate
            Poutput = sigmoid( v W3 + b3 + state * p2)
            output_new = tanh(state) * Poutput
        

        :param iW: weights for cells taking input from preceeding layer.
        Size (4, -1, size)
        :param lW: Weights for connections within layer
        Size (4, size, size )
        :param b: Bias weights for cells taking input from preceeding layer.
        Size (4, size)
        :param p: Weights for peep-holes
        Size (3, size)
        """
        assert len(iW.shape) == 3 and iW.shape[0] == 4
        size = self.size = iW.shape[2]
        assert lW.shape == (4, size, size)
        if b is None:
            b = np.zeros((4, size), dtype=tang_nn_type)
        assert b.shape == (4, size)
        if p is None:
            p = np.zeros((3, size), dtype=tang_nn_type)
        assert p.shape == (3, size)

        self.W = np.ascontiguousarray(np.concatenate((iW, lW), axis=1))
        self.b = np.ascontiguousarray(b)
        self.p = np.ascontiguousarray(p)
        self.isize = iW.shape[1]

    def in_size(self):
        return self.isize

    def out_size(self):
        return self.size

    def run(self, inMat):
        assert self.in_size() == inMat.shape[1]

        out = np.zeros((inMat.shape[0], self.out_size()), dtype=tang_nn_type)
        state = np.zeros(self.out_size(), dtype=tang_nn_type)
        out_prev = np.zeros(self.out_size(), dtype=tang_nn_type)

        for i, v in enumerate(inMat):
            v2 = np.concatenate((v, out_prev))
            #  Forget gate activation
            state *= sigmoid( v2.dot(self.W[2]) + self.b[2] + state * self.p[1] )
            state += tanh(v2.dot(self.W[0]) + self.b[0]) * sigmoid( v2.dot(self.W[1]) + self.b[1] + state * self.p[0])
            #  Output gate activation
            out[i] = tanh(state) * sigmoid(v2.dot(self.W[3]) + self.b[3]  + state * self.p[2])
            out_prev = out[i]
        return out

class reverse:
    """  Runs a recurrent layer in reverse time (backwards)
    """
    def __init__(self, layer):
       self.layer = layer

    def in_size(self):
        return self.layer.in_size()

    def out_size(self):
        return self.layer.out_size()

    def run(self, inMat):
        assert self.in_size() == inMat.shape[1]
        return self.layer.run(inMat[::-1])[::-1]

class parallel:
    """ Run multiple layers in parallel (all have same input and outputs are concatenated)
    """
    def __init__(self, layers):
        in_size = layers[0].in_size()
        for i in range(1, len(layers)):
            assert in_size == layers[i].in_size(), "Incompatible shapes: {} -> {} in layers {}.\n".format(in_size, layers[i].in_size(), i)
        self.layers = layers

    def in_size(self):
        return self.layers[0].in_size()

    def out_size(self):
        return sum(map(lambda x: x.out_size(), self.layers))

    def run(self, inMat):
        assert self.in_size() == inMat.shape[1]
        return np.hstack(map(lambda x: x.run(inMat), self.layers))

class serial:
    """ Run multiple layers serially: output of a layer is the input for the next layer
    """
    def __init__(self, layers, meta=None):
        prev_out_size = layers[0].out_size()
        for i in range(1, len(layers)):
            assert prev_out_size == layers[i].in_size(), "Incompatible shapes: {} -> {} in layers {}.\n".format(prev_out_size, layers[i].in_size(), i)
            prev_out_size = layers[i].out_size()
        self.layers = layers
        self.meta = meta

    def in_size(self):
        return self.layers[0].in_size()

    def out_size(self):
        return self.layers[-1].out_size()

    def run(self, inMat):
        assert self.in_size() == inMat.shape[1]
        tmp = inMat
        for layer in self.layers:
            tmp = layer.run(tmp)
        return tmp

def birnn(layer1, layer2):
    """  Creates a bidirectional RNN from two RNNs
    """
    return parallel([layer1, reverse(layer2)])

