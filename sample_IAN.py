
import argparse
import imp
import time
import logging
import itertools
import os

import numpy as np
from path import Path
import theano
import theano.tensor as T
from theano.tensor.opt import register_canonicalize
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from lasagne.layers import SliceLayer as SL


import GANcheckpoints
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fuel.datasets import CelebA
from discgen_utils import plot_image_grid



## Utilities:
# to_tanh: transforms an array in the range [0,255] to the range [-1,1]
# from_tanh: transforms an array in the range [-1,1] to the range[0,255]
def to_tanh(input):
    return 2.0*(input/255.0)-1.0
    # return input/255.0
 
def from_tanh(input):
    return 255.0*(input+1)/2.0
    # return 255.0*input


### Make Training Functions Method
# This function defines and compiles the computational graphs that define the training, validation, and test functions.
  
def make_training_functions(cfg,model):
    
    # Define input tensors
    # Tensor axes are batch-channel-dim1-dim2
    
    # Image Input 
    X = T.TensorType('float32', [False]*4)('X')
    
    # Latent Input, for providing latent values from the main function
    Z = T.TensorType('float32', [False]*2)('Z') # Latents
    
    # Input layer
    l_in = model['l_in']
    
    # Output layer
    l_out = model['l_out']
    
    # Latent Layer
    l_Z = model['l_Z']
    
    # IAF latent layer:
    l_Z_IAF = model['l_Z_IAF']
    
    # Means
    l_mu = model['l_mu']
    
    # Log-sigmas
    l_ls = model['l_ls']
    
    # IAF Means
    l_IAF_mu = model['l_IAF_mu']
    
    # IAF logsigmas
    l_IAF_ls = model['l_IAF_ls']
    
    # Introspective loss layers
    l_introspect = model['l_introspect']
    
    # Adversarial Discriminator
    l_discrim = model['l_discrim']

    # Sample function
    sample = theano.function([Z],lasagne.layers.get_output(l_out,{l_Z_IAF:Z},deterministic=True),on_unused_input='warn')
    
    sampleZ= theano.function([Z],lasagne.layers.get_output(l_out,{l_Z:Z},deterministic=True),on_unused_input='warn')
    
    # Inference Function--Infer non-IAF_latents given an input X    
    Zfn = theano.function([X],lasagne.layers.get_output(l_Z_IAF,{l_in:X},deterministic=True),on_unused_input='warn')
    
    # IAF function--Infer IAF latents given a latent input Z
    Z_IAF_fn = theano.function([Z],lasagne.layers.get_output(l_Z,{l_Z_IAF:Z},deterministic=True),on_unused_input='warn')


    # Dictionary of Theano Functions
    # tfuncs = {'update_iter':update_iter,
    tfuncs = {'sample': sample,
             'sampleZ': sampleZ,
             'Zfn' : Zfn,
             'Z_IAF_fn': Z_IAF_fn
            }
            
    # Dictionary of Theano Variables        
    tvars = {'X' : X,
             'Z' : Z}
            
    return tfuncs, tvars, model

# Data Loading Function
#
# This function interfaces with a Fuel dataset and returns numpy arrays containing the requested data
def data_loader(cfg,set,offset=0,shuffle=False,seed=42):

    # Define chunk size
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    
    np.random.seed(seed)
    index = np.random.permutation(set.num_examples-offset) if shuffle else np.asarray(range(set.num_examples-offset))
    
    # Open Dataset
    set.open()

    
    # Loop across all data
    for i in xrange(set.num_examples//chunk_size):
        yield to_tanh(np.float32(set.get_data(request = list(index[range(offset+chunk_size*i,offset+chunk_size*(i+1))]))[0]))
    
    # Close dataset
    set.close(state=None)
        

# Main Function
def main(args):

    # Load Config Module from source file
    config_module = imp.load_source('config', args.config_path)
    
    # Get configuration parameters
    cfg = config_module.cfg
   
    # Define name of npz file to which the model parameters will be saved
    weights_fname = str(args.config_path)[:-3]+'.npz'
    
    model = config_module.get_model(interp=False)
    print('Compiling theano functions...')
    
    # Compile functions
    tfuncs, tvars,model = make_training_functions(cfg,model)

    # Test set for interpolations
    test_set = CelebA('64',('test',),sources=('features',))    

    # Loop across epochs
    offset = True
    params = list(set(lasagne.layers.get_all_params(model['l_out'],trainable=True)+\
                              lasagne.layers.get_all_params(model['l_discrim'],trainable=True)+\
                              [x for x in lasagne.layers.get_all_params(model['l_out'])+\
                                lasagne.layers.get_all_params(model['l_discrim']) if x.name[-4:]=='mean' or x.name[-7:]=='inv_std']))
    metadata = GANcheckpoints.load_weights(weights_fname, params)
    epoch = args.epoch if args.epoch>0 else metadata['epoch'] if 'epoch' in metadata else 0
    print('loading weights, epoch is '+str(epoch))

    model['l_IAF_mu'].reset("Once")
    model['l_IAF_ls'].reset("Once")                        
    
    # Open Test Set
    test_set.open()
            
    np.random.seed(epoch*42+5)
    # Generate Random Samples, averaging latent vectors across masks   
    samples = np.uint8(from_tanh(tfuncs['sample'](np.random.randn(27,cfg['num_latents']).astype(np.float32))))
    
    
    np.random.seed(epoch*42+5)
    # Get Reconstruction/Interpolation Endpoints
    endpoints = np.uint8(test_set.get_data(request = list(np.random.choice(test_set.num_examples,6,replace=False)))[0])
   
    # Get reconstruction latents
    Ze = np.asarray(tfuncs['Zfn'](to_tanh(np.float32(endpoints))))
               
    # Get Interpolant Latents
    Z = np.asarray([Ze[2 * i, :] * (1 - j) + Ze[2 * i + 1, :] * j  for i in range(3) for j in [x/6.0 for x in range(7)]],dtype=np.float32)
    
    # Get all images
    images = np.append(samples,np.concatenate([np.insert(endpoints[2*i:2*(i+1),:,:,:],1,np.uint8(from_tanh(tfuncs['sample'](Z[7*i:7*(i+1),:]))),axis=0) for i in range(3)],axis=0),axis=0)


    # Plot images
    plot_image_grid(images,6,9,'pics/'+str(args.config_path)[:-3]+'_sample'+str(epoch)+'.png')
    
    # Close test set
    test_set.close(state=None)
 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='config .py file')
    parser.add_argument('--epoch',type=int,default=0)
    args = parser.parse_args()
    main(args)
