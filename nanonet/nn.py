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

    def run(self, inMat, ctx=None, queueList=None):
        if not queueList:
            assert self.in_size() == inMat.shape[1]
            return self.f(inMat.dot(self.W) + self.b)
        else:
            for mat in inMat:
                assert self.in_size() == mat.shape[1]
            iter = len(inMat)
            
            fp_type = np.float32 # uses this floating point type in the kernel
            kernel_src = kernel_code_feedforward
            
            # Calculate work items 
            local_x = 256
            local_y = 1
            global_x_list = []
            for mat in inMat:
                global_x = mat.shape[0]
                if global_x % local_x:
                    global_x = (global_x / local_x + 1) * local_x 
                global_x_list.append(global_x)
            global_y = 1
            
            # Build the kernel (builds for the first time, then uses cached version)
            opencl_fptype = "double"
            opencl_fptype_suffix = ""
            if fp_type == np.float32:
                opencl_fptype = "float"
                opencl_fptype_suffix = "f"
            opencl_fptype_define = "-DFPTYPE="+opencl_fptype+" -DF="+opencl_fptype_suffix
            prg = cl.Program(ctx, kernel_src).build("-I. -Werror " + opencl_fptype_define + " -DWORK_ITEMS="+str(local_x)+" -DIN_MAT_Y="+str(inMat[0].shape[1]))
            
            inMatcList = []
            for x in xrange(iter):
                inMatcList.append(inMat[x])
            Wc = np.transpose(self.W)
            bc = self.b
            # Convert arrays if the types are different
            if tang_nn_type != fp_type:
                for x in xrange(iter):
                    inMatcList[x] = inMatcList[x].astype(np.float32)
                Wc = Wc.astype(np.float32)
                bc = self.b.astype(np.float32)
            
            # Allocate OpenCL buffers    
            cl_inMatList = []
            for x in xrange(iter):
                cl_inMatList.append(cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(inMatcList[x])))
            cl_W = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(Wc))
            cl_b = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(bc))
            cl_outList = []
            for x in xrange(iter):
                cl_outList.append(cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, inMat[x].shape[0]*self.W.shape[1]*inMatcList[x].itemsize))

            # Run the kernel
            for x in xrange(iter):
                prg.run_layer(queueList[x], (global_x_list[x], global_y), (local_x, local_y), np.int32(inMat[x].shape[0]), np.int32(Wc.shape[0]), cl_inMatList[x], cl_W, cl_b, cl_outList[x])
                queueList[x].flush()
            
            # Copy results back to host (blocking call)
            outList = []
            for x in xrange(iter):
                outList.append(np.zeros((inMat[x].shape[0],self.W.shape[1]), dtype=np.float64))
                if tang_nn_type != fp_type:
                    outRavel = np.ravel(outList[x])
                    outRavel32 = np.zeros(outRavel.size, dtype=np.float32)
                    cl.enqueue_copy(queueList[x], outRavel32, cl_outList[x])
                    outRavel[:] = outRavel32[:]
                else:
                    cl.enqueue_copy(queueList[x], outList[x], cl_outList[x])
            return outList
            
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

    def run(self, inMat, ctx=None, queueList=None):
        if not queueList:
            assert self.in_size() == inMat.shape[1]
            tmp =  inMat.dot(self.W) + self.b
            m = np.amax(tmp, axis=1).reshape((-1,1))
            tmp = np.exp(tmp - m)
            x = np.sum(tmp, axis=1)
            tmp /= x.reshape((-1,1))
            return tmp
        else:
            for mat in inMat:
                assert self.in_size() == mat.shape[1]
            iter = len(inMat)
            
            fp_type = np.float32 # uses this floating point type in the kernel
            kernel_src = kernel_code_softmax
            
            # Calculate work items 
            local_x = 256
            local_y = 1
            global_x_list = []
            for mat in inMat:
                global_x = mat.shape[0]
                if global_x % local_x:
                    global_x = (global_x / local_x + 1) * local_x 
                global_x_list.append(global_x) 
            global_y = 1
            local_x_softmax = 256
            
            # Build the kernel (builds for the first time, then uses cached version)
            opencl_fptype = "double"
            opencl_fptype_suffix = ""
            if fp_type == np.float32:
                opencl_fptype = "float"
                opencl_fptype_suffix = "f"
            opencl_fptype_define = "-DFPTYPE="+opencl_fptype+" -DF="+opencl_fptype_suffix
            prg = cl.Program(ctx, kernel_src).build("-I. -Werror " + opencl_fptype_define + 
                " -DWORK_ITEMS="+str(local_x)+" -DIN_MAT_Y="+str(inMat[0].shape[1]) + 
                " -DWORK_ITEMS_PAR="+str(local_x_softmax) + " -DITER="+str((self.W.shape[1]-1)/local_x_softmax))
            
            inMatcList = []
            for x in xrange(iter):
                inMatcList.append(inMat[x])
            Wc = np.transpose(self.W)
            bc = self.b
            # Convert arrays if the types are different
            if tang_nn_type != fp_type:
                for x in xrange(iter):
                    inMatcList[x] = inMatcList[x].astype(np.float32)
                Wc = Wc.astype(np.float32)
                bc = self.b.astype(np.float32)
            
            # Allocate OpenCL buffers    
            cl_inMatList = []
            for x in xrange(iter):
                cl_inMatList.append(cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(inMatcList[x])))
            cl_W = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(Wc))
            cl_b = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(bc))
            cl_outList = []
            for x in xrange(iter):
                cl_outList.append(cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, inMat[x].shape[0]*self.W.shape[1]*inMatcList[x].itemsize))

            # Run the kernel
            for x in xrange(iter):
                prg.run_layer(queueList[x], (global_x_list[x], global_y), (local_x, local_y), np.int32(inMat[x].shape[0]), np.int32(Wc.shape[0]), cl_inMatList[x], cl_W, cl_b, cl_outList[x])
                queueList[x].flush()
            for x in xrange(iter):
                prg.run_softmax(queueList[x], (inMat[x].shape[0]*local_x_softmax, 1), (local_x_softmax, 1), np.int32(Wc.shape[0]), cl_outList[x])
                queueList[x].flush()
            
            # Copy results back to host (blocking call)
            outList = []
            for x in xrange(iter):
                outList.append(np.zeros((inMat[x].shape[0],self.W.shape[1]), dtype=np.float64))
                if tang_nn_type != fp_type:
                    outRavel = np.ravel(outList[x])
                    outRavel32 = np.zeros(outRavel.size, dtype=np.float32)
                    cl.enqueue_copy(queueList[x], outRavel32, cl_outList[x])
                    outRavel[:] = outRavel32[:]
                else:
                    cl.enqueue_copy(queueList[x], outList[x], cl_outList[x])
            return outList


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

    def run(self, inMat, ctx=None, queueList=None):
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

    def run(self, inMat, ctx=None, queueList=None):
        if not queueList:
            assert self.in_size() == inMat.shape[1]
            out = np.zeros((inMat.shape[0], self.out_size()), dtype=tang_nn_type)
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
        else:
            for mat in inMat:
                assert self.in_size() == mat.shape[1]
            iter = len(inMat)
            is_nvidia = True if "nvidia" in ctx.get_info(cl.context_info.DEVICES)[0].get_info(cl.device_info.VENDOR).lower() else False
            
            outList = []
            for x in xrange(iter):
                outList.append(np.zeros((inMat[x].shape[0], self.out_size()), dtype=tang_nn_type))
            
            fp_type = np.float32 # uses this floating point type in the kernel
            kernel_src = kernel_code_lstm
            
            # Build the kernel (builds for the first time, then uses cached version)
            opencl_fptype = "double"
            opencl_fptype_suffix = ""
            if fp_type == np.float32:
                opencl_fptype = "float"
                opencl_fptype_suffix = "f"
            opencl_fptype_define = "-DFPTYPE="+opencl_fptype+" -DF="+opencl_fptype_suffix
            is_nvidia_define = ""
            if is_nvidia:
                is_nvidia_define = " -DNVIDIA"
            prg = cl.Program(ctx, kernel_src).build("-I. -Werror " + opencl_fptype_define + " -DWORK_ITEMS="+str(self.out_size())+
                " -DOUT_SIZE="+str(self.out_size())+" -DIN_MAT_Y="+str(inMat[x].shape[1])+is_nvidia_define)
            
            inMatcList = []
            for x in xrange(iter):
                inMatcList.append(inMat[x])
            if is_nvidia:
                Wc = np.transpose(self.W, axes=[1,0,2])
            else:
                Wc = np.transpose(self.W, axes=[0,2,1])
            bc = self.b
            pc = self.p
            # Convert arrays if the types are different
            if tang_nn_type != fp_type:
                for x in xrange(iter):
                    inMatcList[x] = inMatcList[x].astype(np.float32)
                Wc = Wc.astype(np.float32)
                bc = self.b.astype(np.float32)
                pc = self.p.astype(np.float32)
            
            # Allocate OpenCL buffers
            cl_inMatList = []
            for x in xrange(iter):
                cl_inMatList.append(cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(inMatcList[x])))
            cl_W = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(Wc))
            cl_b = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(bc))
            cl_p = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(pc))
            cl_outList = []
            for x in xrange(iter):
                cl_outList.append(cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, outList[x].shape[0]*outList[x].shape[1]*inMatcList[x].itemsize))

            # Run the kernel
            for x in xrange(iter):
                prg.run_lstm_layer(queueList[x], (self.out_size(), 1), (self.out_size(), 1), np.int32(inMatcList[x].shape[0]), np.int32(Wc.shape[1]), np.int32(Wc.shape[2]), cl_inMatList[x], cl_W, cl_b, cl_p, cl_outList[x])
                queueList[x].flush()
            
            # Copy results back to host (blocking call)
            for x in xrange(iter):
                outRavel = np.ravel(outList[x])
                if tang_nn_type != fp_type:
                    outRavel32 = np.zeros(outRavel.size, dtype=np.float32)
                    cl.enqueue_copy(queueList[x], outRavel32, cl_outList[x])
                    outRavel[:] = outRavel32[:]
                else:
                    cl.enqueue_copy(queueList[x], outRavel, cl_outList[x])
                outList[x] = np.copy(np.reshape(outRavel, (outList[x].shape[0], outList[x].shape[1])))
            return outList


class reverse:
    """  Runs a recurrent layer in reverse time (backwards)
    """
    def __init__(self, layer):
       self.layer = layer

    def in_size(self):
        return self.layer.in_size()

    def out_size(self):
        return self.layer.out_size()

    def run(self, inMat, ctx=None, queueList=None):
        if not queueList:
            assert self.in_size() == inMat.shape[1]
            return self.layer.run(inMat[::-1])[::-1]
        else:
            inMatList = []
            for mat in inMat:
                assert self.in_size() == mat.shape[1]
                inMatList.append(mat[::-1])            
            postList= self.layer.run(inMatList, ctx, queueList)
            postListTmp = []
            for post in postList:
                postListTmp.append(post[::-1])
            return postListTmp 

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

    def run(self, inMat, ctx=None, queueList=None):
        if not queueList:
            assert self.in_size() == inMat.shape[1]
            return np.hstack(map(lambda x: x.run(inMat), self.layers))
        else:
            for mat in inMat:
                assert self.in_size() == mat.shape[1]
            tmp = map(lambda x: x.run(inMat, ctx, queueList), self.layers)
            tmp2 = map(list, zip(*tmp))
            tmp3 = []
            for t in tmp2:
                tmp3.append(np.hstack(t))
            return tmp3 

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

    def run(self, inMat, ctx=None, queueList=None):
        if not queueList:
            assert self.in_size() == inMat.shape[1]
            tmp = inMat
            for layer in self.layers:
                tmp = layer.run(tmp)
            return tmp
        else:
            for mat in inMat:
                assert self.in_size() == mat.shape[1]
            tmp = inMat
            for layer in self.layers:
                tmp = layer.run(tmp, ctx, queueList)
            return tmp

def birnn(layer1, layer2):
    """  Creates a bidirectional RNN from two RNNs
    """
    return parallel([layer1, reverse(layer2)])

kernel_code_feedforward = """
#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel __attribute__((reqd_work_group_size(WORK_ITEMS, 1, 1)))
__kernel void run_layer(
    int inMatx,
    int Wx, 
    __global const FPTYPE* restrict inMat, 
    __global const FPTYPE* restrict W, 
    __global const FPTYPE* restrict b, 
    __global FPTYPE* restrict ret
){
    int id = get_global_id(0);
    if(id < inMatx)
    {
        FPTYPE inMatBuffer[IN_MAT_Y];
        for(int z = 0; z < IN_MAT_Y; ++z)
            inMatBuffer[z] = inMat[id*IN_MAT_Y+z];
        
        for(int y = 0; y < Wx; ++y)
        {
            FPTYPE r = 0.0F;
            for(int z = 0; z < IN_MAT_Y; ++z)
                r += inMatBuffer[z] * W[y*IN_MAT_Y+z];
            ret[id*Wx+y] = tanh(r + b[y]);
        }
    }
}
"""

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
    __local FPTYPE v2[V2_SIZE];
    FPTYPE state = 0.0F;
    FPTYPE r[4];
    FPTYPE bb[4];
    FPTYPE pp[3];
    
    for(int y = 0; y < 4; ++y)
        bb[y] = b[y*OUT_SIZE+id]; 
        
    for(int y = 0; y < 3; ++y)
        pp[y] = p[y*OUT_SIZE+id];
        
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
#ifdef NVIDIA
            r[0] += v * Wtr[v2x*Wtry*Wtrz+(0*Wtrz)+id];
            r[1] += v * Wtr[v2x*Wtry*Wtrz+(1*Wtrz)+id];  
            r[2] += v * Wtr[v2x*Wtry*Wtrz+(2*Wtrz)+id];
            r[3] += v * Wtr[v2x*Wtry*Wtrz+(3*Wtrz)+id];
#else
            r[0] += v * Wtr[0*Wtry*Wtrz+(id*Wtrz)+v2x];
            r[1] += v * Wtr[1*Wtry*Wtrz+(id*Wtrz)+v2x];
            r[2] += v * Wtr[2*Wtry*Wtrz+(id*Wtrz)+v2x];
            r[3] += v * Wtr[3*Wtry*Wtrz+(id*Wtrz)+v2x];
#endif
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

kernel_code_softmax = """
#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel __attribute__((reqd_work_group_size(WORK_ITEMS, 1, 1)))
void run_layer(
    int inMatx,
    int Wx, 
    __global const FPTYPE* restrict inMat, 
    __global const FPTYPE* restrict W, 
    __global const FPTYPE* restrict b, 
    __global FPTYPE* restrict ret
){
    int id = get_global_id(0);
    if(id < inMatx)
    {
        FPTYPE inMatBuffer[IN_MAT_Y];
        for(int z = 0; z < IN_MAT_Y; ++z)
            inMatBuffer[z] = inMat[id*IN_MAT_Y+z];
        
        for(int y = 0; y < Wx; ++y)
        {
            FPTYPE r = 0.0F;
            for(int z = 0; z < IN_MAT_Y; ++z)
                r += inMatBuffer[z] * W[y*IN_MAT_Y+z];
            ret[id*Wx+y] = r + b[y];
        }
    }
}

inline void parallel_sum(__local FPTYPE * restrict buffer)
{
    // Perform parallel reduction
    int local_index = get_local_id(0);
    for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) 
    {
        if (local_index < offset) 
            buffer[local_index] += buffer[local_index + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void parallel_max(__local FPTYPE * restrict buffer)
{
    // Perform parallel reduction
    int local_index = get_local_id(0);
    for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) 
    {
        if (local_index < offset)
            if (buffer[local_index + offset] > buffer[local_index]) 
                buffer[local_index] = buffer[local_index + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel __attribute__((reqd_work_group_size(WORK_ITEMS_PAR, 1, 1)))       // WORK_ITEMS_PAR = 2^n
void run_softmax(
    int size,                         // 2^n+1
    __global FPTYPE * restrict inout
){
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);
    
    __local FPTYPE buffer[WORK_ITEMS_PAR];
    __local FPTYPE last_in;
    FPTYPE elem[ITER];
    FPTYPE max = 0.0F;
    FPTYPE sum = 0.0F;
    
    if(local_id == 0)
        last_in = inout[(group_id+1) * size - 1]; // access last size-1 element
    barrier(CLK_LOCAL_MEM_FENCE);
    max = last_in;
    
    for(int x = 0; x < size-1; x += local_size)
    {
        elem[x/local_size] = buffer[local_id] = inout[group_id * size + local_id + x];
        barrier(CLK_LOCAL_MEM_FENCE);
        parallel_max(buffer);
        max = max > buffer[0] ? max : buffer[0];
    }
    
    for(int x = 0; x < size-1; x += local_size)
        elem[x/local_size] = exp(elem[x/local_size] - max); 

    if(local_id == 0)
        last_in = exp(last_in-max);

    for(int x = 0; x < size-1; x += local_size)
    {
        buffer[local_id] = elem[x/local_size];
        barrier(CLK_LOCAL_MEM_FENCE);
        parallel_sum(buffer);
        sum += buffer[0];
    }
    sum += last_in;
    
    for(int x = 0; x < size-1; x += local_size)
        inout[group_id * size + local_id + x] = elem[x/local_size] / sum;
    if(local_id == 0)
        inout[(group_id+1) * size - 1] = last_in / sum;  
}
""" 