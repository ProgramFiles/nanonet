import abc
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_approx(x):
    """ Pade approximation of tanh function
    http://musicdsp.org/archive.php?classid=5#238
    """
    xsqr = np.square(x)
    tanh_p = x * (27.0 + xsqr) / (27.0 + 9.0 * xsqr)
    return np.clip(tanh_p, -1.0, 1.0)

def sigmoid(x):
    return np.reciprocal(1.0 + np.exp(-x))

def sigmoid_approx(x):
   """ Approximation of sigmoid function
   https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L217
   """
   xabs = np.fabs(x)
   tmp = np.where(xabs < 3.0, 0.4677045353015495 + 0.02294064733985825 * (xabs - 1.7), 0.497527376843365)
   tmp = np.where(xabs < 1.7, 0.75 * xabs / (1.0 + xabs), tmp)
   return np.sign(x) * tmp + 0.5

def linear(x):
    return x

def softplus(x):
    return np.log1p(np.exp(x))

def relu(x):
    return np.where(x > 0.0, x, 0.0)

"""  Convention: inMat row major (C ordering) as (time, state)
"""
dtype = np.float64

class Layer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(self, inMat):
        """  Run network layer
        """
        return

class RNN(Layer):

    @abc.abstractmethod
    def step(self, in_vec, state):
        """ A single step along the RNN
        :param in_vec: Input to node
        :param state: Hidden state from previous node
        """
        return


class feedforward(Layer):
    """  Basic feedforward layer
         out = f( inMat W + b )

    :param W: Weight matrix of dimension (|input|, size)
    :param b: Bias vector of length  size.  Optional with default of no bias.
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, W, b=None, fun=tanh):
        assert b is None or len(b) == W.shape[1]
        self.b = np.zeros(W.shape[1], dtype=dtype) if b is None else b
        self.W = W
        self.f = fun

    def in_size(self):
        return self.W.shape[0]

    def out_size(self):
        return self.W.shape[1]

    def run(self, inMat):
        assert self.in_size() == inMat.shape[1]
        return self.f(inMat.dot(self.W) + self.b)

class softmax(Layer):
    """  Softmax layer
         tmp = exp( inmat W + b )
         out = row_normalise( tmp )

    :param W: Weight matrix of dimension (|input|, size)
    :param b: Bias vector of length size.  Optional with default of no bias.
    """
    def __init__(self, W, b=None):
        assert b is None or len(b) == W.shape[1]
        self.b = np.zeros(W.shape[1], dtype=dtype) if b is None else b
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

class rnn(RNN):
    """ A simple recurrent layer
        Step:  state_new = fun( [state_old, input_new] W + b )
               output_new = state_new

    :param W: Weight matrix of dimension (|input| + size, size)
    :param b: Bias vector of length  size.  Optional with default of no bias.
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, W, b=None, fun=tanh):
        assert W.shape[0] > W.shape[1]
        assert b is None or len(b) == W.shape[1]
        self.b = np.zeros(W.shape[1], dtype=dtype) if b is None else b
        self.W = W

        self.fun = fun
        self.size = W.shape[0] - W.shape[1]

    def in_size(self):
        return self.size

    def out_size(self):
        return self.W.shape[1]

    def step(self, in_vec, state):
        state_out = self.fun(np.concatenate((state, in_vec)).dot(self.W) + self.b)
        return state_out

    def run(self, inMat):
        assert self.in_size() == inMat.shape[1]
        out = np.zeros((inMat.shape[0], self.out_size()), dtype=dtype)
        state = np.zeros(self.out_size(), dtype=dtype)
        for i, v in enumerate(inMat):
            state = self.step(v, state)
            out[i] = state
        return out

class lstm(RNN):
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
            b = np.zeros((4, size), dtype=dtype)
        assert b.shape == (4, size)
        if p is None:
            p = np.zeros((3, size), dtype=dtype)
        assert p.shape == (3, size)

        self.iW = np.ascontiguousarray(iW.transpose((1,0,2)).reshape((-1, 4 * size)))
        self.lW = np.ascontiguousarray(lW.transpose((1,0,2)).reshape((size, 4 * size)))
        self.b = np.ascontiguousarray(b).reshape(-1)
        self.p = np.ascontiguousarray(p)
        self.isize = iW.shape[1]

    def in_size(self):
        return self.isize

    def out_size(self):
        return self.size

    def step(self, in_vec, in_state):
        vW = in_vec.dot(self.iW)
        out_prev = in_state[:self.size]
        state = in_state[self.size:]
        outW = out_prev.dot(self.lW)
        sumW = vW + outW  + self.b
        sumW = sumW.reshape((4, self.size))

        #  Forget gate activation
        state *= sigmoid(sumW[2] + state * self.p[1] )
        #  Update state with input
        state += tanh(sumW[0]) * sigmoid(sumW[1] + state * self.p[0])
        #  Output gate activation
        out = tanh(state) * sigmoid(sumW[3]  + state * self.p[2])
        return np.concatenate((out, state))


    def run(self, inMat):
        assert self.in_size() == inMat.shape[1]

        out = np.zeros((inMat.shape[0], self.out_size()), dtype=dtype)
        state = np.zeros(2 * self.out_size(), dtype=dtype)

        for i, v in enumerate(inMat):
            state = self.step(v, state)
            out[i] = state
        return out

class reverse(Layer):
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

class parallel(Layer):
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

class serial(Layer):
    """ Run multiple layers serially: output of a layer is the input for the next layer
    """
    def __init__(self, layers):
        prev_out_size = layers[0].out_size()
        for i in range(1, len(layers)):
            assert prev_out_size == layers[i].in_size(), "Incompatible shapes: {} -> {} in layers {}.\n".format(prev_out_size, layers[i].in_size(), i)
            prev_out_size = layers[i].out_size()
        self.layers = layers

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
