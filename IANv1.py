## IAN with binary adversarial loss and no orthogonal regularization

import lasagne
import lasagne.layers
import lasagne.layers.dnn
from lasagne.layers import SliceLayer as SL
from lasagne.layers import batch_norm as BN
from lasagne.layers import ElemwiseSumLayer as ESL
from lasagne.layers import ElemwiseMergeLayer as EML
from lasagne.layers import NonlinearityLayer as NL
from lasagne.layers import DenseLayer as DL
from lasagne.layers import Upscale2DLayer
from lasagne.init import Normal as initmethod
from lasagne.init import Orthogonal
from lasagne.nonlinearities import elu
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as lrelu
from lasagne.nonlinearities import sigmoid
from lasagne.layers.dnn import Conv2DDNNLayer as C2D
from lasagne.layers.dnn import Pool2DDNNLayer as P2D
from lasagne.layers import TransposedConv2DLayer as TC2D
from lasagne.layers import ConcatLayer as CL
import numpy as np
import theano.tensor as T
import theano
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from math import sqrt



from layers import MDBLOCK, DeconvLayer, MinibatchLayer, beta_layer, MADE, IAFLayer, GaussianSampleLayer, MDCL

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
lr_schedule = { 0: 0.0002,25:0.0001,50:0.00005,75:0.00001}
cfg = {'batch_size' : 16,
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
       'batches_per_chunk': 64,
       'max_epochs' :150,
       'checkpoint_every_nth' : 1,
       'num_latents': 100,
       'recon_weight': 3.0,
       'feature_weight': 1.0,
       'dg_weight': 1.0,
       'dd_weight':1.0,
       'agr_weight':1.0,
       'ags_weight':1.0, 
       'n_shuffles' : 1,
       }
       
def get_model(interp=False):
    dims, n_channels = tuple(cfg['dims']), cfg['n_channels']
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
        name =  'enc_conv4'
        ),name = 'bnorm4')  
    
  
    print(lasagne.layers.get_output_shape(l_enc_conv4,(196,3,64,64))) 
    l_enc_fc1 = BN(DL(
        incoming = l_enc_conv4,
        num_units = 1000,
        W = initmethod(0.02),
        nonlinearity = relu,
        name =  'enc_fc1'
        ),
        name = 'bnorm_enc_fc1')

    # Define latent values  
    l_enc_mu,l_enc_logsigma = [BN(DL(incoming = l_enc_fc1,num_units=cfg['num_latents'],nonlinearity = None,name='enc_mu'),name='mu_bnorm'),
                               BN(DL(incoming = l_enc_fc1,num_units=cfg['num_latents'],nonlinearity = None,name='enc_logsigma'),name='ls_bnorm')]
    l_Z_IAF = GaussianSampleLayer(l_enc_mu, l_enc_logsigma, name='l_Z_IAF')
    l_IAF_mu,l_IAF_logsigma = [MADE(l_Z_IAF,[cfg['num_latents']],'l_IAF_mu'),MADE(l_Z_IAF,[cfg['num_latents']],'l_IAF_ls')]
    l_Z = IAFLayer(l_Z_IAF,l_IAF_mu,l_IAF_logsigma,name='l_Z')
    l_dec_fc2 = DL(
        incoming = l_Z,
        num_units = 1024*16,
        nonlinearity = None,
        W=initmethod(0.02),
        name='l_dec_fc2')
    l_unflatten = lasagne.layers.ReshapeLayer(
        incoming = l_dec_fc2,
        shape = ([0],1024,4,4),
        )
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
        
    l_dec_conv4 = BN(DeconvLayer(
        incoming = l_dec_conv3,
        num_filters = 64,
        filter_size = [5,5],
        stride = [2,2],
        crop = (2,2),
        W = initmethod(0.02),
        nonlinearity = relu,
        name =  'dec_conv4'
        ),name = 'bnorm_dc4') 
        
    R = NL(MDCL(l_dec_conv4,    
            num_filters=2,
            scales = [2,3,4],
            name = 'R'),sigmoid)
    G = NL(ESL([MDCL(l_dec_conv4,    
        num_filters=2,
        scales = [2,3,4],
        name =  'G_a'
        ),
        MDCL(R,    
            num_filters=2,
            scales = [2,3,4],
            name =  'G_b'
        )]),sigmoid)
    B = NL(ESL([MDCL(l_dec_conv4,    
            num_filters=2,
            scales = [2,3,4],
            name =  'B_a'
        ),
            MDCL(CL([R,G]),    
            num_filters=2,
            scales = [2,3,4],
            name =  'B_b'
        )]),sigmoid)  
    l_out=CL([beta_layer(SL(R,slice(0,1),1),SL(R,slice(1,2),1)),beta_layer(SL(G,slice(0,1),1),SL(G,slice(1,2),1)),beta_layer(SL(B,slice(0,1),1),SL(B,slice(1,2),1))])   
   
    minibatch_discrim =  MinibatchLayer(lasagne.layers.GlobalPoolLayer(l_enc_conv4), num_kernels=500,name='minibatch_discrim')    
    l_discrim = DL(incoming = minibatch_discrim,
        num_units = 1,
        nonlinearity = lasagne.nonlinearities.sigmoid,
        b = None,
        W=initmethod(0.02),
        name = 'discrimi')
        
        
    return {'l_in':l_in, 
            'l_out':l_out,
            'l_mu':l_enc_mu,
            'l_ls':l_enc_logsigma,            
            'l_Z':l_Z,
            'l_IAF_mu': l_IAF_mu,
            'l_IAF_ls': l_IAF_logsigma,
            'l_Z_IAF': l_Z_IAF,
            'l_introspect':[l_enc_conv1, l_enc_conv2,l_enc_conv3,l_enc_conv4],

            'l_discrim' : l_discrim}

            