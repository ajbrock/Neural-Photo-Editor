### Simple IAN model for use with Neural Photo Editor
# This model is a simplified version of the Introspective Adversarial Network that does not
# make use of Multiscale Dilated Convolutional blocks, Ternary Adversarial Loss, or an
# autoregressive RGB-Beta layer. It's designed to be sleeker and to run on laptop GPUs with <1GB of memory.

import numpy as np

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

import theano.tensor as T

from math import sqrt


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import GaussianSampleLayer,MinibatchLayer 
lr_schedule = { 0: 0.0002}
cfg = {'batch_size' : 128,
       'learning_rate' : lr_schedule,
       'optimizer' : 'Adam',
       'beta1' : 0.5,
       'update_ratio' : 1,
       'decay_rate' : 0,
       'reg' : 1e-5,
       'momentum' : 0.9,
       'shuffle' : True,
       'dims' : (64,64),
       'n_channels' : 3,
       'n_classes' : 10,
       'batches_per_chunk': 64,
       'max_epochs' :250,
       'checkpoint_every_nth' : 1,
       'num_latents': 100,
       'recon_weight': 3.0,
       'feature_weight': 1.0,
       }
       
      

    
def get_model(dnn=True):
    if dnn:
        import lasagne.layers.dnn
        from lasagne.layers.dnn import Conv2DDNNLayer as C2D
        from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
        from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
        from layers import DeconvLayer
    else:
        import lasagne.layers
        from lasagne.layers import Conv2DLayer as C2D
    
    dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
    shape = (None, n_channels)+dims
    l_in = lasagne.layers.InputLayer(shape=shape)
    l_enc_conv1 = C2D(
        incoming = l_in,
        num_filters = 128,
        filter_size = [5,5],
        stride = [2,2],
        pad = (2,2),
        W = initmethod(0.02),
        nonlinearity = lrelu(0.2),
        flip_filters=False,
        name =  'enc_conv1'
        )
    l_enc_conv2 = BN(C2D(
        incoming = l_enc_conv1,
        num_filters = 256,
        filter_size = [5,5],
        stride = [2,2],
        pad = (2,2),
        W = initmethod(0.02),
        nonlinearity = lrelu(0.2),
        flip_filters=False,
        name =  'enc_conv2'
        ),name = 'bnorm2')
    l_enc_conv3 = BN(C2D(
        incoming = l_enc_conv2,
        num_filters = 512,
        filter_size = [5,5],
        stride = [2,2],
        pad = (2,2),
        W = initmethod(0.02),
        nonlinearity = lrelu(0.2),
        flip_filters=False,
        name =  'enc_conv3'
        ),name = 'bnorm3')
    l_enc_conv4 = BN(C2D(
        incoming = l_enc_conv3,
        num_filters = 1024,
        filter_size = [5,5],
        stride = [2,2],
        pad = (2,2),
        W = initmethod(0.02),
        nonlinearity = lrelu(0.2),
        flip_filters=False,
        name =  'enc_conv4'
        ),name = 'bnorm4')         
    l_enc_fc1 = BN(DL(
        incoming = l_enc_conv4,
        num_units = 1000,
        W = initmethod(0.02),
        nonlinearity = elu,
        name =  'enc_fc1'
        ),
        name = 'bnorm_enc_fc1')
    l_enc_mu,l_enc_logsigma = [BN(DL(incoming = l_enc_fc1,num_units=cfg['num_latents'],nonlinearity = None,name='enc_mu'),name='mu_bnorm'),
                               BN(DL(incoming = l_enc_fc1,num_units=cfg['num_latents'],nonlinearity = None,name='enc_logsigma'),name='ls_bnorm')]

    l_Z = GaussianSampleLayer(l_enc_mu, l_enc_logsigma, name='l_Z')
    l_dec_fc2 = BN(DL(
        incoming = l_Z,
        num_units = 1024*16,
        nonlinearity = relu,
        W=initmethod(0.02),
        name='l_dec_fc2'),
        name = 'bnorm_dec_fc2') 
    l_unflatten = lasagne.layers.ReshapeLayer(
        incoming = l_dec_fc2,
        shape = ([0],1024,4,4),
        )
    if dnn:
        l_dec_conv1 = BN(DeconvLayer(
            incoming = l_unflatten,
            num_filters = 512,
            filter_size = [5,5],
            stride = [2,2],
            crop = (2,2),
            W = initmethod(0.02),
            nonlinearity = relu,
            name =  'dec_conv1'
            ),name = 'bnorm_dc1')
        l_dec_conv2 = BN(DeconvLayer(
            incoming = l_dec_conv1,
            num_filters = 256,
            filter_size = [5,5],
            stride = [2,2],
            crop = (2,2),
            W = initmethod(0.02),
            nonlinearity = relu,
            name =  'dec_conv2'
            ),name = 'bnorm_dc2')
        l_dec_conv3 = BN(DeconvLayer(
            incoming = l_dec_conv2,
            num_filters = 128,
            filter_size = [5,5],
            stride = [2,2],
            crop = (2,2),
            W = initmethod(0.02),
            nonlinearity = relu,
            name =  'dec_conv3'
            ),name = 'bnorm_dc3')
        l_out = DeconvLayer(
            incoming = l_dec_conv3,
            num_filters = 3,
            filter_size = [5,5],
            stride = [2,2],
            crop = (2,2),
            W = initmethod(0.02),
            b = None,
            nonlinearity = lasagne.nonlinearities.tanh,
            name =  'dec_out'
            )
    else:    
        l_dec_conv1 = SL(SL(BN(TC2D(
            incoming = l_unflatten,
            num_filters = 512,
            filter_size = [5,5],
            stride = [2,2],
            crop = (1,1),
            W = initmethod(0.02),
            nonlinearity = relu,
            name =  'dec_conv1'
            ),name = 'bnorm_dc1'),indices=slice(1,None),axis=2),indices=slice(1,None),axis=3)
        l_dec_conv2 = SL(SL(BN(TC2D(
            incoming = l_dec_conv1,
            num_filters = 256,
            filter_size = [5,5],
            stride = [2,2],
            crop = (1,1),
            W = initmethod(0.02),
            nonlinearity = relu,
            name =  'dec_conv2'
            ),name = 'bnorm_dc2'),indices=slice(1,None),axis=2),indices=slice(1,None),axis=3)
        l_dec_conv3 = SL(SL(BN(TC2D(
            incoming = l_dec_conv2,
            num_filters = 128,
            filter_size = [5,5],
            stride = [2,2],
            crop = (1,1),
            W = initmethod(0.02),
            nonlinearity = relu,
            name =  'dec_conv3'
            ),name = 'bnorm_dc3'),indices=slice(1,None),axis=2),indices=slice(1,None),axis=3)
        l_out = SL(SL(TC2D(
            incoming = l_dec_conv3,
            num_filters = 3,
            filter_size = [5,5],
            stride = [2,2],
            crop = (1,1),
            W = initmethod(0.02),
            b = None,
            nonlinearity = lasagne.nonlinearities.tanh,
            name =  'dec_out'
            ),indices=slice(1,None),axis=2),indices=slice(1,None),axis=3)
# l_in,num_filters=1,filter_size=[5,5],stride=[2,2],crop=[1,1],W=dc.W,b=None,nonlinearity=None)
    minibatch_discrim =  MinibatchLayer(lasagne.layers.GlobalPoolLayer(l_enc_conv4), num_kernels=500,name='minibatch_discrim')    
    l_discrim = DL(incoming = minibatch_discrim, 
        num_units = 1,
        nonlinearity = lasagne.nonlinearities.sigmoid,
        b = None,
        W=initmethod(),
        name = 'discrimi')
        

        
    return {'l_in':l_in, 
            'l_out':l_out,
            'l_mu':l_enc_mu,
            'l_ls':l_enc_logsigma,            
            'l_Z':l_Z,
            'l_introspect':[l_enc_conv1, l_enc_conv2,l_enc_conv3,l_enc_conv4],
            'l_discrim' : l_discrim}

            
