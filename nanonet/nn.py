import pyopencl as cl
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
use_opencl = True
ctx = None
queue = None

def init_opencl():
    global ctx
    if ctx != None:
        return
    ctx = cl.create_some_context()
    global queue
    queue = cl.CommandQueue(ctx)


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
        if use_opencl == True:
            # Init OpenCL (does it once)
            init_opencl()
            global ctx
            global queue
            
            fp_type = np.float32 # uses this floating point type in the kernel
            kernel_src = kernel_code_lstm
            
            # Build the kernel (builds for the first time, then uses cached version)
            opencl_fptype = "double"
            opencl_fptype_suffix = ""
            if fp_type == np.float32:
                opencl_fptype = "float"
                opencl_fptype_suffix = "f"
            opencl_fptype_define = "-DFPTYPE="+opencl_fptype+" -DF="+opencl_fptype_suffix
            prg = cl.Program(ctx, kernel_src).build("-I. -Werror " + opencl_fptype_define + " -DWORK_ITEMS="+str(self.out_size())+" -DOUT_SIZE="+str(self.out_size())+" -DIN_MAT_Y="+str(inMat.shape[1]))
            
            inMatc = inMat
            Wc = np.transpose(self.W, axes=[0,2,1])
            bc = self.b
            pc = self.p
            # Convert arrays if the types are different
            if tang_nn_type != fp_type:
                inMatc = inMat.astype(np.float32)
                Wc = Wc.astype(np.float32)
                bc = self.b.astype(np.float32)
                pc = self.p.astype(np.float32)
            
            # Allocate OpenCL buffers    
            cl_inMat = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(inMatc))
            cl_W = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(Wc))
            cl_b = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(bc))
            cl_p = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(pc))
            cl_out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.shape[0]*out.shape[1]*inMatc.itemsize)

            # Run the kernel
            prg.run_lstm_layer(queue, (self.out_size(), 1), (self.out_size(), 1), np.int32(inMatc.shape[0]), np.int32(Wc.shape[1]), np.int32(Wc.shape[2]), cl_inMat, cl_W, cl_b, cl_p, cl_out)
            
            # Copy results back to host (blocking call)
            outRavel = np.ravel(out)
            if tang_nn_type != fp_type:
                outRavel32 = np.zeros(outRavel.size, dtype=np.float32)
                cl.enqueue_copy(queue, outRavel32, cl_out)
                outRavel[:] = outRavel32[:]
            else:
                cl.enqueue_copy(queue, outRavel, cl_out)
            out = np.copy(np.reshape(outRavel, (out.shape[0], out.shape[1])))
        else:
            state = np.zeros(self.out_size(), dtype=tang_nn_type)
            out_prev = np.zeros(self.out_size(), dtype=tang_nn_type)
    
            for i, v in enumerate(inMat):
                v2 = np.concatenate((v, out_prev))
                #  Forget gate activation
                state *= sigmoid( v2.dot(self.W[2]) + self.b[2] + state * self.p[1] )
                #  Update state with input
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

kernel_code_lstm = """
#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#define V2_SIZE (IN_MAT_Y + OUT_SIZE)

__kernel __attribute__((reqd_work_group_size(WORK_ITEMS, 1, 1))) 
void run_lstm_layer(
    int inMatx,
    int Wtry, 
    int Wtrz,
    __global const FPTYPE* restrict inMat, 
    __global const FPTYPE* restrict Wtr, 
    __global const FPTYPE* restrict b, 
    __global const FPTYPE* restrict p,
    __global FPTYPE* restrict out
) {
    int id = get_global_id(0);
    __local FPTYPE v2[IN_MAT_Y + OUT_SIZE];
    FPTYPE state = 0.0F;
    FPTYPE r[4];
    FPTYPE bb[4];
    FPTYPE pp[3];
    FPTYPE W[4][V2_SIZE];
    
    for(int y = 0; y < 4; ++y)
        bb[y] = b[y*OUT_SIZE+id]; 
        
    for(int y = 0; y < 3; ++y)
        pp[y] = p[y*OUT_SIZE+id];
        
    for(int y = 0; y < 4; ++y)
        for(int v2x = 0; v2x < V2_SIZE; ++v2x)
            W[y][v2x] = Wtr[y*Wtry*Wtrz+(id*Wtrz)+v2x];
    
    v2[IN_MAT_Y+id] = 0.0F;
    for(int x = 0; x < inMatx; ++x)
    {
        if(id < IN_MAT_Y)
          v2[id] = inMat[x*IN_MAT_Y+id];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int y = 0; y < 4; ++y)
            r[y] = 0.0F;
        for(int v2x = 0; v2x < V2_SIZE; ++v2x)
        {
            const FPTYPE v = v2[v2x];
            r[0] += v * W[0][v2x];
            r[1] += v * W[1][v2x];  
            r[2] += v * W[2][v2x];
            r[3] += v * W[3][v2x];
        }
        
        // Forget gate activation
        FPTYPE tmp = r[2] + bb[2] + state * pp[1];
        FPTYPE sigm = 1.0F/(1.0F + exp(-tmp));
        state *= sigm;
          
        tmp = r[1] + bb[1] + state * pp[0];
        sigm = 1.0F/(1.0F + exp(-tmp));
          
        tmp = tanh(r[0] + bb[0]);
        state += tmp * sigm;
          
        //  Output gate activation
        tmp = r[3] + bb[3] + state * pp[2];
        sigm = 1.0F/(1.0F + exp(-tmp));
        v2[IN_MAT_Y+id] = tanh(state) * sigm; 
        out[x*OUT_SIZE+id] = v2[IN_MAT_Y+id];
        barrier(CLK_LOCAL_MEM_FENCE); 
    }      
}
"""
