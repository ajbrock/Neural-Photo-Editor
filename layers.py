### Custom Lasagne Layers for Introspective Adversarial Networks
# A Brock, 2016
#
# Layers that are not my own creation should be appropriately attributed here
# MADE wrapped from the implementation by M. Germain et al: https://github.com/mgermain/MADE
# Gaussian Sample layer from Tencia Lee's Recipe: https://github.com/Lasagne/Recipes/blob/master/examples/variational_autoencoder/variational_autoencoder.py
# Minibatch Discrimination layer from OpenAI's Improved GAN Techniques: https://github.com/openai/improved-gan
# Deconv Layer adapted from Radford's DCGAN: https://github.com/Newmu/dcgan_code

from __future__ import division
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers

from lasagne.layers import SliceLayer as SL
from lasagne.layers import batch_norm as BN
from lasagne.layers import ElemwiseSumLayer as ESL
from lasagne.layers import NonlinearityLayer as NL
from lasagne.layers import DenseLayer as DL
from lasagne.init import Normal as initmethod
from lasagne.nonlinearities import elu
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as lrelu

from lasagne.layers import TransposedConv2DLayer as TC2D
from lasagne.layers import ConcatLayer as CL


from math import sqrt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from mask_generator import MaskGenerator

# BatchReNorm Layer using cuDNN's BatchNorm
# This layer implements BatchReNorm (https://arxiv.org/abs/1702.03275), 
# which modifies BatchNorm to include running-average statistics in addition to 
# per-batch statistics in a well-principled manner. The RMAX and DMAX parameters 
# are clip parameters which should be fscalars that you'll need to manage in the training
# loop, as they follow an annealing schedule (an example of which is given in the paper).
# I've been adjusting this schedule based on the total number of iterations relative
# to the number given in the paper, so for a ~50,000 iteration training run, I anneal
# RMAX between 1k and 5k iterations rather than 5k and 25k. 

# NOTE: Ideally you should not have to manage RMAX and DMAX separately, so
# if someone wants to write a default_update similar to the one used for
# running_average and running_inv_std, that would be excellent.
class BatchReNormDNNLayer(BatchNormLayer):
    
    def __init__(self, incoming, RMAX,DMAX,axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
                 mean=lasagne.init.Constant(0), inv_std=lasagne.init.Constant(1), **kwargs):
        super(BatchReNormDNNLayer, self).__init__(
                incoming, axes, epsilon, alpha, beta, gamma, mean, inv_std,
                **kwargs)
        all_but_second_axis = (0,) + tuple(range(2, len(self.input_shape)))
        
        self.RMAX,self.DMAX = RMAX,DMAX
        
        if self.axes not in ((0,), all_but_second_axis):
            raise ValueError("BatchNormDNNLayer only supports normalization "
                             "across the first axis, or across all but the "
                             "second axis, got axes=%r" % (axes,))

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
           
        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]
        # and prepare the converse pattern removing those broadcastable axes
        unpattern = [d for d in range(input.ndim) if d not in self.axes]

        # call cuDNN if needed, obtaining normalized outputs and statistics
        if not use_averages or update_averages:
            # cuDNN requires beta/gamma tensors; create them if needed
            shape = tuple(s for (d, s) in enumerate(input.shape)
                          if d not in self.axes)
            gamma = self.gamma or theano.tensor.ones(shape)
            beta = self.beta or theano.tensor.zeros(shape)
            mode = 'per-activation' if self.axes == (0,) else 'spatial'
            
            (normalized,
             input_mean,
             input_inv_std) = theano.sandbox.cuda.dnn.dnn_batch_normalization_train(
                    input, gamma.dimshuffle(pattern), beta.dimshuffle(pattern),
                    mode, self.epsilon)

        # normalize with stored averages, if needed
        if use_averages:
            mean = self.mean.dimshuffle(pattern)
            inv_std = self.inv_std.dimshuffle(pattern)
            gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
            beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
            normalized = (input - mean) * (gamma * inv_std) + beta

        # update stored averages, if needed
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean.dimshuffle(unpattern))
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std.dimshuffle(unpattern))
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            # dummy = running_mean + running_inv_std).dimshuffle(pattern)
            r = T.clip(running_inv_std.dimshuffle(pattern)/input_inv_std,1/self.RMAX,self.RMAX)
            d = T.clip( (input_mean-running_mean.dimshuffle(pattern))*running_inv_std.dimshuffle(pattern),-self.DMAX,self.DMAX)
            normalized = normalized * r + d

        return normalized



# More Efficient MDCL layer
# When seeking to construct an MDC block, drop this into a Conv2D layer instead; it's faster.
# You can also easily re-parameterize this to a full-rank MDC block by dropping in the
# line that creates baseW into the for loop such that a new W is sampled each time.
def mdclW(num_filters,num_channels,filter_size,winit,name,scales):
    # Coefficient Initializer
    sinit = lasagne.init.Constant(1.0/(1+len(scales)))
    # Total filter size
    size = filter_size + (filter_size-1)*(scales[-1]-1)
    # Multiscale Dilated Filter 
    W = T.zeros((num_filters,num_channels,size,size))
    # Undilated Base Filter
    baseW = theano.shared(lasagne.utils.floatX(winit.sample((num_filters,num_channels,filter_size,filter_size))),name=name+'.W')
    for scale in enumerate(scales[::-1]): # enumerate backwards so that we place the main filter on top
            W = T.set_subtensor(W[:,:,scales[-1]-scale:size-scales[-1]+scale:scale,scales[-1]-scale:size-scales[-1]+scale:scale],
                                  baseW*theano.shared(lasagne.utils.floatX(sinit.sample(num_filters)), name+'.coeff_'+str(scale)).dimshuffle(0,'x','x','x'))
    return W

# Subpixel Upsample Layer from (https://arxiv.org/abs/1609.05158)
# This layer uses a set of r^2 set_subtensor calls to reorganize the tensor in a subpixel-layer upscaling style
# as done in the ESPCN Magic ony paper for super-resolution.
# r is the upscale factor.
# c is the number of output channels.
class SubpixelLayer(lasagne.layers.Layer):
    def __init__(self, incoming,r,c, **kwargs):
        super(SubpixelLayer, self).__init__(incoming, **kwargs)
        self.r=r # Upscale factor
        self.c=c # number of output channels
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.c,self.r*input_shape[2],self.r*input_shape[3])

    def get_output_for(self, input, deterministic=False, **kwargs):
        out = T.zeros((input.shape[0],self.output_shape[1],self.output_shape[2],self.output_shape[3]))
        for x in xrange(self.r): # loop across all feature maps belonging to this channel
            for y in xrange(self.r):
                out=T.set_subtensor(out[:,:,x::self.r,y::self.r],input[:,self.r*x+y::self.r*self.r,:,:])
        return out
# Subpixel Upsample Layer using reshapes as in https://github.com/Tetrachrome/subpixel. This implementation appears to be 10x slower than
# the set_subtensor implementation, presumably because of the extra reshapes after the splits.         
class SubpixelLayer2(lasagne.layers.Layer):
    def __init__(self, incoming,r,c, **kwargs):
        super(SubpixelLayer2, self).__init__(incoming, **kwargs)
        self.r=r
        self.c=c

    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.c,self.r*input_shape[2],self.r*input_shape[3])

    def get_output_for(self, input, deterministic=False, **kwargs):
        def _phase_shift(input,r):
            bsize,c,a,b = input.shape[0],1,self.output_shape[2]//r,self.output_shape[3]//r
            X = T.reshape(input, (bsize,r,r,a,b))
            X = T.transpose(X, (0, 3,4,1,2))  # bsize, a, b, r2,r1
            X = T.split(x=X,splits_size=[1]*a,n_splits=a,axis=1)  # a, [bsize, b, r, r]
            X = [T.reshape(x,(bsize,b,r,r))for x in X]
            X = T.concatenate(X,axis=2)  # bsize, b, a*r, r 
            X = T.split(x=X,splits_size =[1]*b,n_splits=b,axis=1)  # b, [bsize, a*r, r]
            X = [T.reshape(x,(bsize,a*r,r))for x in X]
            X = T.concatenate(X,axis=2) # bsize, a*r, b*r 
            return X.dimshuffle(0,'x',1,2)
        Xc = T.split(x=input,splits_size =[input.shape[1]//self.c]*self.c,n_splits=self.c,axis=1)
        return T.concatenate([_phase_shift(xc,self.r) for xc in Xc],axis=1)        

# Multiscale Dilated Convolution Block
# This function (not a layer in and of itself, though you could make it one) returns a set of concatenated conv2d and dilatedconv2d layers.
# Each layer uses the same basic filter W, operating at a different dilation factor (or taken as the mean of W for the 1x1 conv).
# The channel-wise output of each layer is weighted by a set of coefficients, which are initialized to 1 / the total number of dilation scales,
# meaning that were starting by taking an elementwise mean. These should be learnable parameters.

# NOTES: - I'm considering changing the variable names to be more descriptive, and look less like ridiculous academic code. It's on the to-do list.
#        - I keep the bias and nonlinearity out of the default definition for this layer, as I expect it to be batchnormed and nonlinearized in the model config.
def MDCL(incoming,num_filters,scales,name,dnn=True):
    if dnn:
        from lasagne.layers.dnn import Conv2DDNNLayer as C2D
    # W initialization method--this should also work as Orthogonal('relu'), but I have yet to validate that as thoroughly.
    winit = initmethod(0.02)
    
    # Initialization method for the coefficients
    sinit = lasagne.init.Constant(1.0/(1+len(scales)))
    
    # Number of incoming channels
    ni =lasagne.layers.get_output_shape(incoming)[1]
    
    # Weight parameter--the primary parameter for this block
    W = theano.shared(lasagne.utils.floatX(winit.sample((num_filters,lasagne.layers.get_output_shape(incoming)[1],3,3))),name=name+'W')
    
    # Primary Convolution Layer--No Dilation
    n = C2D(incoming = incoming,
                            num_filters = num_filters,
                            filter_size = [3,3],
                            stride = [1,1],
                            pad = (1,1),
                            W = W*theano.shared(lasagne.utils.floatX(sinit.sample(num_filters)), name+'_coeff_base').dimshuffle(0,'x','x','x'), # Note the broadcasting dimshuffle for the num_filter scalars.
                            b = None,
                            nonlinearity = None,
                            name = name+'base'
                        )
    # List of remaining layers. This should probably just all be concatenated into a single list rather than being a separate deal.
    nd = []    
    for i,scale in enumerate(scales):
        
        # I don't think 0 dilation is technically defined (or if it is it's just the regular filter) but I use it here as a convenient keyword to grab the 1x1 mean conv.
        if scale==0:
            nd.append(C2D(incoming = incoming,
                            num_filters = num_filters,
                            filter_size = [1,1],
                            stride = [1,1],
                            pad = (0,0),
                            W = T.mean(W,axis=[2,3]).dimshuffle(0,1,'x','x')*theano.shared(lasagne.utils.floatX(sinit.sample(num_filters)), name+'_coeff_1x1').dimshuffle(0,'x','x','x'),
                            b = None,
                            nonlinearity = None,
                            name = name+str(scale)))
        # Note the dimshuffles in this layer--these are critical as the current DilatedConv2D implementation uses a backward pass.
        else:
            nd.append(lasagne.layers.DilatedConv2DLayer(incoming = lasagne.layers.PadLayer(incoming = incoming, width=(scale,scale)),
                                num_filters = num_filters,
                                filter_size = [3,3],
                                dilation=(scale,scale),
                                W = W.dimshuffle(1,0,2,3)*theano.shared(lasagne.utils.floatX(sinit.sample(num_filters)), name+'_coeff_'+str(scale)).dimshuffle('x',0,'x','x'),
                                b = None,
                                nonlinearity = None,
                                name =  name+str(scale)))
    return ESL(nd+[n])

# MDC-based Upsample Layer.
# This is a prototype I don't make use of extensively. It's operational but it doesn't seem to improve results yet.
def USL(incoming,num_filters,scales,name,dnn=True):
    if dnn:
        from lasagne.layers.dnn import Conv2DDNNLayer as C2D
    
    # W initialization method--this should also work as Orthogonal('relu'), but I have yet to validate that as thoroughly.
    winit = initmethod(0.02)
    
    # Initialization method for the coefficients
    sinit = lasagne.init.Constant(1.0/(1+len(scales)))
    
    # Number of incoming channels
    ni =lasagne.layers.get_output_shape(incoming)[1]
    
    # Weight parameter--the primary parameter for this block
    W = theano.shared(lasagne.utils.floatX(winit.sample((num_filters,lasagne.layers.get_output_shape(incoming)[1],3,3))),name=name+'W')
    
    # Primary Convolution Layer--No Dilation
    n = C2D(incoming = Upscale2DLayer(incoming,2),
                            num_filters = num_filters,
                            filter_size = [3,3],
                            stride = [1,1],
                            pad = (1,1),
                            W = W*theano.shared(lasagne.utils.floatX(sinit.sample(num_filters)), name+'_coeff_base').dimshuffle(0,'x','x','x'),
                            b = None,
                            nonlinearity = None,
                            name = name+'base'
                        )
    # Remaining layers              
    nd = []    
    for i,scale in enumerate(scales):                    
        if scale==0:
            nd.append(C2D(incoming = Upscale2DLayer(incoming,2),
                            num_filters = num_filters,
                            filter_size = [1,1],
                            stride = [1,1],
                            pad = (0,0),
                            W = T.mean(W,axis=[2,3]).dimshuffle(0,1,'x','x')*theano.shared(lasagne.utils.floatX(sinit.sample(num_filters)), name+'_coeff_1x1').dimshuffle(0,'x','x','x'),
                            b = None,
                            nonlinearity = None,
                            name = name+'1x1'
                        ))
        else:
            nd.append(lasagne.layers.DilatedConv2DLayer(incoming = lasagne.layers.PadLayer(incoming = Upscale2DLayer(incoming,2), width=(scale,scale)),
                                num_filters = num_filters,
                                filter_size = [3,3],
                                dilation=(scale,scale),
                                W = W.dimshuffle(1,0,2,3)*theano.shared(lasagne.utils.floatX(sinit.sample(num_filters)), name+'_coeff_'+str(scale)).dimshuffle('x',0,'x','x'),
                                b = None,
                                nonlinearity = None,
                                name =  name+str(scale)))
    
    # A single deconv layer is also concatenated here. Like I said, it's a prototype!
    nd.append(DeconvLayer(incoming = incoming,
                            num_filters = num_filters,
                            filter_size = [3,3],
                            stride = [2,2],
                            crop = (1,1),
                            W = W.dimshuffle(1,0,2,3)*theano.shared(lasagne.utils.floatX(sinit.sample(num_filters)), name+'_coeff_deconv').dimshuffle('x',0,'x','x'),
                            b = None,
                            nonlinearity = None,
                            name = name+'deconv'
                        ))

    return ESL(nd+[n])     

#MDC-based Downsample Layer.
# This is a prototype I don't make use of extensively. It's operational and it seems like it works alright, but it's restrictively expensive
# and I am not PARALLELICUS, god of GPUs, so I don't have the memory to spare for it.   
# Note that this layer does not currently support having a 0 scale like the others do, and just has a 1x1-stride2 conv by default.
def DSL(incoming,num_filters,scales,name,dnn=True):
    if dnn:
        from lasagne.layers.dnn import Conv2DDNNLayer as C2D
    # W initialization method--this should also work as Orthogonal('relu'), but I have yet to validate that as thoroughly.
    winit = initmethod(0.02)
    
    # Initialization method for the coefficients
    sinit = lasagne.init.Constant(1.0/(1+len(scales)))
    
    # Number of incoming channels
    ni =lasagne.layers.get_output_shape(incoming)[1]
    
    # Weight parameter--the primary parameter for this block
    W = theano.shared(lasagne.utils.floatX(winit.sample((num_filters,lasagne.layers.get_output_shape(incoming)[1],3,3))),name=name+'W')
    
    # Main layer--3x3 conv with stride 2
    n = C2D(incoming = incoming,
                            num_filters = num_filters,
                            filter_size = [3,3],
                            stride = [2,2],
                            pad = (1,1),
                            W = W*theano.shared(lasagne.utils.floatX(sinit.sample(num_filters)), name+'_coeff_base').dimshuffle(0,'x','x','x'),
                            b = None,
                            nonlinearity = None,
                            name = name+'base'
                        )

                      
    nd = []    
    for i,scale in enumerate(scales):

        p = P2D(incoming = incoming,
                                    pool_size = scale,
                                    stride = 2,
                                    pad = (1,1) if i else (0,0),
                                    mode = 'average_exc_pad',
                                    )

        nd.append(C2D(incoming = p,
                    num_filters = num_filters,
                    filter_size = [3,3],
                    stride = (1,1),
                    pad = (1,1),
                    W = W*theano.shared(lasagne.utils.floatX(sinit.sample(num_filters)), name+'_coeff_'+str(scale)).dimshuffle(0,'x','x','x'),#.dimshuffle('x',0),
                    b = None,
                    nonlinearity = None,
                    name =  name+str(scale)))            
                                                  
        
    nd.append(C2D(incoming = incoming,
                            num_filters = num_filters,
                            filter_size = [1,1],
                            stride = [2,2],
                            pad = (0,0),
                            W = T.mean(W,axis=[2,3]).dimshuffle(0,1,'x','x')*theano.shared(lasagne.utils.floatX(sinit.sample(num_filters)), name+'_coeff_1x1').dimshuffle(0,'x','x','x'),
                            b = None,
                            nonlinearity = None,
                            name = name+'1x1'
                        ))
   
    return ESL(nd+[n])    

# Beta Distribution Layer   
# This layer takes in a batch_size batch, 2-channel, NxN dimension layer and returns the output of the first channel
# divided by the sum of both channels, which is equivalent to finding the expected value for a beta distribution.
# Note that this version of the layer scales to {-1,1} for compatibility with tanh.
class beta_layer(lasagne.layers.MergeLayer):
    def __init__(self, alpha,beta, **kwargs):
        super(beta_layer, self).__init__([alpha,beta], **kwargs)

    def get_output_shape_for(self, input_shape):
        print(input_shape)
        return input_shape[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        alpha,beta = inputs
        # return 2*T.true_div(alpha,T.add(alpha,beta)+1e-8)-1
        return 2*(alpha/(alpha+beta+1e-8))-1

# Convenience Function to produce a residual pre-activation MDCL block        
def MDBLOCK(incoming,num_filters,scales,name,nonlinearity):
    return NL(BN(ESL([incoming,
         MDCL(NL(BN(MDCL(NL(BN(incoming,name=name+'bnorm0'),nonlinearity),num_filters,scales,name),name=name+'bnorm1'),nonlinearity),
              num_filters,
              scales,
              name+'2')]),name=name+'bnorm2'),nonlinearity)  
              
# Gaussian Sample Layer for VAE from Tencia Lee
class GaussianSampleLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)

# DeconvLayer adapted from Radford's DCGAN Implementation
class DeconvLayer(lasagne.layers.conv.BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 crop=0, untie_biases=False,
                 W=initmethod(), b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False,
                 **kwargs):
        super(DeconvLayer, self).__init__(
                incoming, num_filters, filter_size, stride, crop, untie_biases,
                W, b, nonlinearity, flip_filters, n=2, **kwargs)
        # rename self.crop to self.pad
        self.crop = self.pad
        del self.pad

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        # first two sizes are swapped compared to a forward convolution
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        
        # when called from the constructor, self.crop is still called self.pad:
        crop = getattr(self, 'crop', getattr(self, 'pad', None))
        crop = crop if isinstance(crop, tuple) else (crop,) * self.n
        batchsize = input_shape[0]
        return(batchsize,self.num_filters)+(input_shape[2]*2,input_shape[3]*2)
        # return ((batchsize, self.num_filters) +
                # tuple(conv_input_length(input, filter, stride, p)
                      # for input, filter, stride, p
                      # in zip(input_shape[2:], self.filter_size,
                             # self.stride, crop)))

    def convolve(self, input, **kwargs):
        
        # Messy to have these imports here, but seems to allow for switching DNN off.
        from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
        from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool 
        # Straight outta Radford
        img = gpu_contiguous(input)
        kerns = gpu_contiguous(self.W)
        desc = GpuDnnConvDesc(border_mode=self.crop, subsample=self.stride,
                              conv_mode='conv')(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*self.stride[0], img.shape[3]*self.stride[1]).shape, kerns.shape)
        out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*self.stride[0], img.shape[3]*self.stride[1])
        conved = GpuDnnConvGradI()(kerns, img, out, desc)

        return conved
        
# Minibatch discrimination layer from OpenAI's improved GAN techniques       
class MinibatchLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_kernels, dim_per_kernel=5, theta=lasagne.init.Normal(0.05),
                 log_weight_scale=lasagne.init.Constant(0.), b=lasagne.init.Constant(-1.), **kwargs):
        super(MinibatchLayer, self).__init__(incoming, **kwargs)
        self.num_kernels = num_kernels
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.theta = self.add_param(theta, (num_inputs, num_kernels, dim_per_kernel), name="theta")
        self.log_weight_scale = self.add_param(log_weight_scale, (num_kernels, dim_per_kernel), name="log_weight_scale")
        self.W = self.theta * (T.exp(self.log_weight_scale)/T.sqrt(T.sum(T.square(self.theta),axis=0))).dimshuffle('x',0,1)
        self.b = self.add_param(b, (num_kernels,), name="b")
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:])+self.num_kernels)

    def get_output_for(self, input, init=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        
        activation = T.tensordot(input, self.W, [[1], [0]])
        abs_dif = (T.sum(abs(activation.dimshuffle(0,1,2,'x') - activation.dimshuffle('x',1,2,0)),axis=2)
                    + 1e6 * T.eye(input.shape[0]).dimshuffle(0,'x',1))

        if init:
            mean_min_abs_dif = 0.5 * T.mean(T.min(abs_dif, axis=2),axis=0)
            abs_dif /= mean_min_abs_dif.dimshuffle('x',0,'x')
            self.init_updates = [(self.log_weight_scale, self.log_weight_scale-T.log(mean_min_abs_dif).dimshuffle(0,'x'))]
        
        f = T.sum(T.exp(-abs_dif),axis=2)

        if init:
            mf = T.mean(f,axis=0)
            f -= mf.dimshuffle('x',0)
            self.init_updates.append((self.b, -mf))
        else:
            f += self.b.dimshuffle('x',0)

        return T.concatenate([input, f], axis=1)  

# Convenience function to define an inception-style block
def InceptionLayer(incoming,param_dict,block_name):
    branch = [0]*len(param_dict)
    # Loop across branches
    for i,dict in enumerate(param_dict):
        for j,style in enumerate(dict['style']): # Loop up branch
            branch[i] = C2D(
                incoming = branch[i] if j else incoming,
                num_filters = dict['num_filters'][j],
                filter_size = dict['filter_size'][j],
                pad =  dict['pad'][j] if 'pad' in dict else None,
                stride = dict['stride'][j],
                W = initmethod('relu'),
                nonlinearity = dict['nonlinearity'][j],
                name = block_name+'_'+str(i)+'_'+str(j)) if style=='convolutional'\
            else NL(lasagne.layers.dnn.Pool2DDNNLayer(
                incoming=incoming if j == 0 else branch[i],
                pool_size = dict['filter_size'][j],
                mode = dict['mode'][j],
                stride = dict['stride'][j],
                pad = dict['pad'][j],
                name = block_name+'_'+str(i)+'_'+str(j)),
                nonlinearity = dict['nonlinearity'][j]) if style=='pool'\
            else lasagne.layers.DilatedConv2DLayer(
                incoming = lasagne.layers.PadLayer(incoming = incoming if j==0 else branch[i],width = dict['pad'][j]) if 'pad' in dict else incoming if j==0 else branch[i],
                num_filters = dict['num_filters'][j],
                filter_size = dict['filter_size'][j],
                dilation = dict['dilation'][j],
                # pad = dict['pad'][j] if 'pad' in dict else None,
                W = initmethod('relu'),
                nonlinearity = dict['nonlinearity'][j],
                name = block_name+'_'+str(i)+'_'+str(j))  if style== 'dilation'\
            else DL(
                    incoming = incoming if j==0 else branch[i],
                    num_units = dict['num_filters'][j],
                    W = initmethod('relu'),
                    b = None,
                    nonlinearity = dict['nonlinearity'][j],
                    name = block_name+'_'+str(i)+'_'+str(j))   
                # Apply Batchnorm    
            branch[i] = BN(branch[i],name = block_name+'_bnorm_'+str(i)+'_'+str(j)) if dict['bnorm'][j] else branch[i]
        # Concatenate Sublayers        
            
    return CL(incomings=branch,name=block_name)

# Convenience function to define an inception-style block with upscaling    
def InceptionUpscaleLayer(incoming,param_dict,block_name):
    branch = [0]*len(param_dict)
    # Loop across branches
    for i,dict in enumerate(param_dict):
        for j,style in enumerate(dict['style']): # Loop up branch
            branch[i] = TC2D(
                incoming = branch[i] if j else incoming,
                num_filters = dict['num_filters'][j],
                filter_size = dict['filter_size'][j],
                crop = dict['pad'][j] if 'pad' in dict else None,
                stride = dict['stride'][j],
                W = initmethod('relu'),
                nonlinearity = dict['nonlinearity'][j],
                name = block_name+'_'+str(i)+'_'+str(j)) if style=='convolutional'\
            else NL(
                    incoming = lasagne.layers.dnn.Pool2DDNNLayer(
                        incoming = lasagne.layers.Upscale2DLayer(
                            incoming=incoming if j == 0 else branch[i],
                            scale_factor = dict['stride'][j]),
                        pool_size = dict['filter_size'][j],
                        stride = [1,1],
                        mode = dict['mode'][j],
                        pad = dict['pad'][j],
                        name = block_name+'_'+str(i)+'_'+str(j)),
                    nonlinearity = dict['nonlinearity'][j])
                # Apply Batchnorm    
            branch[i] = BN(branch[i],name = block_name+'_bnorm_'+str(i)+'_'+str(j)) if dict['bnorm'][j] else branch[i]
        # Concatenate Sublayers        
            
    return CL(incomings=branch,name=block_name)

# Convenience function to efficiently generate param dictionaries for use with InceptioNlayer
def pd(num_layers=2,num_filters=32,filter_size=(3,3),pad=1,stride = (1,1),nonlinearity=elu,style='convolutional',bnorm=1,**kwargs):
    input_args = locals()    
    input_args.pop('num_layers')
    return {key:entry if type(entry) is list else [entry]*num_layers for key,entry in input_args.iteritems()}  

# Possible Conv2DDNN convenience function. Remember to delete the C2D import at the top if you use this    
# def C2D(incoming = None, num_filters = 32, filter_size= [3,3],pad = 'same',stride = [1,1], W = initmethod('relu'),nonlinearity = elu,name = None):
    # return lasagne.layers.dnn.Conv2DDNNLayer(incoming,num_filters,filter_size,stride,pad,False,W,None,nonlinearity,False)

# Shape-Preserving Gaussian Sample layer for latent vectors with spatial dimensions.
# This is a holdover from an "old" (i.e. I abandoned it last month) idea. 
class GSL(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
        super(GSL, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shape):
        print(input_shape)
        return input_shape[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(logsigma.shape)

# Convenience function to return list of sampled latent layers
def GL(mu,ls):
    return([GSL(z_mu,z_ls) for z_mu,z_ls in zip(mu,ls)])

# Convenience function to return a residual layer. It's not really that much more convenient than ESL'ing,
# but I like being able to see when I'm using Residual connections as opposed to Elemwise-sums    
def ResLayer(incoming, IB,nonlinearity):
    return NL(ESL([IB,incoming]),nonlinearity)


# Inverse autoregressive flow layer       
class IAFLayer(lasagne.layers.MergeLayer):
    def __init__(self, z, mu, logsigma, **kwargs):
        super(IAFLayer, self).__init__([z,mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        z,mu, logsigma = inputs
        return (z - mu) / T.exp(logsigma)

# Masked layer for MADE, adopted from M.Germain        
class MaskedLayer(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, mask_generator,layerIdx,W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(MaskedLayer, self).__init__(incoming, num_units, W,b, nonlinearity,**kwargs)
        self.mask_generator = mask_generator
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.weights_mask = self.add_param(spec = np.ones((num_inputs, num_units),dtype=np.float32),
                                           shape = (num_inputs, num_units),
                                           name='weights_mask',
                                           trainable=False,
                                           regularizable=False)
        self.layerIdx = layerIdx
        self.shuffle_update = [(self.weights_mask, mask_generator.get_mask_layer_UPDATE(self.layerIdx))]
   
    def get_output_for(self,input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        activation = T.dot(input, self.W*self.weights_mask)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

        
# Stripped-Down Direct Input masked layer: Combine this with ESL and a masked layer to get a true DIML.
# Consider making this a simultaneous subclass of MaskedLayer and elemwise sum layer for cleanliness
#  adopted from M.Germain  
class DIML(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, mask_generator,layerIdx,W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=None,**kwargs):
        super(DIML, self).__init__(incoming, num_units, W,b, nonlinearity,**kwargs)
        
        self.mask_generator = mask_generator
        self.layerIdx = layerIdx
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.weights_mask = self.add_param(spec = np.ones((num_inputs, num_units),dtype=np.float32),
                                           shape = (num_inputs, num_units),
                                           name='weights_mask',
                                           trainable=False,
                                           regularizable=False)
        

        self.shuffle_update = [(self.weights_mask, self.mask_generator.get_direct_input_mask_layer_UPDATE(self.layerIdx + 1))]         

   
    def get_output_for(self,input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        
        activation = T.dot(input, self.W*self.weights_mask)            
        
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)        

# Conditioning Masked Layer 
# Currently not used.       
# class CML(MaskedLayer):

    # def __init__(self, incoming, num_units, mask_generator,use_cond_mask=False,U=lasagne.init.GlorotUniform(),W=lasagne.init.GlorotUniform(),
                 # b=init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        # super(CML, self).__init__(incoming, num_units, mask_generator,W,
                 # b, nonlinearity,**kwargs)
        
        # self.use_cond_mask=use_cond_mask
        # if use_cond_mask:            
            # self.U = self.add_param(spec = U,
                                    # shape = (num_inputs, num_units),
                                    # name='U',
                                    # trainable=True,
                                    # regularizable=False)theano.shared(value=self.weights_initialization((self.n_in, self.n_out)), name=self.name+'U', borrow=True)
            # self.add_param(self.U,name = 
    # def get_output_for(self,input,**kwargs):
       # lin = self.lin_output = T.dot(input, self.W * self.weights_mask) + self.b  
       # if self.use_cond_mask:
           # lin = lin+T.dot(T.ones_like(input), self.U * self.weights_mask)
       # return lin if self._activation is None else self._activation(lin) 


       
# Made layer, adopted from M.Germain        
class MADE(lasagne.layers.Layer):
    def __init__(self,z,hidden_sizes,name,nonlinearity=lasagne.nonlinearities.rectify,output_nonlinearity=None, **kwargs):
        # self.rng = rng if rng else RandomStreams(lasagne.random.get_rng().randint(1234))
        super(MADE, self).__init__(z, **kwargs)
        
        # Incoming latents
        self.z = z
        
        # List defining hidden units in each layer
        self.hidden_sizes = hidden_sizes
        
        # Layer name for saving parameters.
        self.name = name
        
        # nonlinearity
        self.nonlinearity = nonlinearity
        
        # Output nonlinearity
        self.output_nonlinearity = output_nonlinearity
        
        # Control parameters from original MADE
        mask_distribution=0
        use_cond_mask = False
        direct_input_connect = "Output"
        direct_output_connect = False
        self.shuffled_once = False
        
        # Mask generator
        self.mask_generator = MaskGenerator(lasagne.layers.get_output_shape(z)[1], hidden_sizes, mask_distribution)
        
        # Build the MADE
        # TODO: Consider making this more compact by directly writing to the layers list
        self.input_layer = MaskedLayer(incoming = z, 
                                  num_units = hidden_sizes[0], 
                                  mask_generator = self.mask_generator,
                                  layerIdx = 0,
                                  W = lasagne.init.Orthogonal('relu'),
                                  nonlinearity=self.nonlinearity,
                                  name = self.name+'_input')
                                  
        self.layers = [self.input_layer]
        
        for i in range(1, len(hidden_sizes)):
        
            self.layers += [MaskedLayer(incoming = self.layers[-1], 
                                       num_units = hidden_sizes[i], 
                                       mask_generator = self.mask_generator,
                                       layerIdx = i,
                                       W = lasagne.init.Orthogonal('relu'),
                                       nonlinearity=self.nonlinearity,
                                       name = self.name+'_layer_'+str(i))]
                                                        
        outputLayerIdx = len(self.layers)
        
        # Output layer
        self.layers += [MaskedLayer(incoming = self.layers[-1], 
                                       num_units = lasagne.layers.get_output_shape(z)[1], 
                                       mask_generator = self.mask_generator,
                                       layerIdx = outputLayerIdx,
                                       W = lasagne.init.Orthogonal('relu'),
                                       nonlinearity = self.output_nonlinearity,
                                       name = self.name+'_output_W'),
                                DIML(incoming = z, 
                                num_units = lasagne.layers.get_output_shape(z)[1],
                                mask_generator = self.mask_generator,
                                layerIdx = outputLayerIdx,
                                W = lasagne.init.Orthogonal('relu'),
                                nonlinearity = self.output_nonlinearity,
                                name = self.name+'_output_D')]



        masks_updates = [layer_mask_update for l in self.layers for layer_mask_update in l.shuffle_update]
        self.update_masks = theano.function(name='update_masks',
                                        inputs=[],
                                        updates=masks_updates)
        # Make the true output layer by ESL'ing the DIML and masked layer
        self.final_layer= ESL([self.layers[-2],self.layers[-1]])
        # self.output_layer = self.layers[-1]
        # params = [p for p in l.get_params(trainable=True) for l in self.layers]
        # print(params)

    def get_output_for(self, input, deterministic=False, **kwargs):
        return lasagne.layers.get_output(self.final_layer,{self.z:input})
    
    def get_params(self, unwrap_shared=True, **tags):
        params = []
        for l in self.layers:
            for p in l.get_params(**tags):
                params.append(p)
        return(params)        
        # params = [p for p in l.get_params(trainable=True) for l in self.layers]
        # return params
        # return [p for p in lay.get_params(unwrap_shared,**tags) for lay in self.layers]
        # return lasagne.layers.get_all_params(self.final_layer,trainable=True)
    
    def shuffle(self, shuffling_type):
        if shuffling_type == "Once" and self.shuffled_once is False:
            self.mask_generator.shuffle_ordering()
            self.mask_generator.sample_connectivity()
            self.update_masks()
            self.shuffled_once = True
            return

        if shuffling_type in ["Ordering", "Full"]:
            self.mask_generator.shuffle_ordering()
        if shuffling_type in ["Connectivity", "Full"]:
            self.mask_generator.sample_connectivity()
        self.update_masks()

    def reset(self, shuffling_type, last_shuffle=0):
        self.mask_generator.reset()

        # Always do a first shuffle so that the natural order does not gives us an edge
        self.shuffle("Full")

        # Set the mask to the requested shuffle
        for i in range(last_shuffle):
            self.shuffle(shuffling_type)    