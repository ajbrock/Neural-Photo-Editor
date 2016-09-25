###
# Hierarchical Adversarial Introspective Autoencoder Main training Function
#
# A Brock 2016
# TO DO: 
# 1. Cleanup
# 2. Split validation/test sets appropriately, do early stopping on validation set, do reconstructions on test set
# 13. Learn to render to HTML with VTK, and do the 2d-manifold thing. Consider rendering directly to a webpage?

# 16. Read and UNDERSTAND Batch Norm and Improved Gan Training papers. Really no excuse for not absolutely understanding these; consider checking WeightNorm (by salimans?) too.

# 19. Add code to automatically test if a saved npz of the weights already exists



# 27. Figure out how to incorporate log likelihood into introspective reconstruction error, along with new log_sigma_thetas

# 32. Do we want the classifier to take in the LATENTS or the layer just before the latents?
# 33. Do we want to propagate the log_sigma through the decoder network somehow to make log-likelihood work better? I'd prefer to use some entropy measure, honestly

# 36. Add support for class-conditional-ness

# 38. Test regime: number of latent variables vs. accuracy

# 40. Multi-GPU? Split a batch across two GPUs? Send the autoencoding batch to one gpu, send the adversarial batch to another
# 42. Get a validation set together
# 
# Consider hacking the adversarial BCE by changing from zeros/ones to -1's and 2's.

# consider only using adversarial gen loss on random samples

# Consider the effects of batch normalization on an X_hat batch split into half-recon and half-generated,
# specifically: 1. Does batch-norm depend on the 0-axis of the tensor at all, such that we need to shuffle that tensor? I don't think it doesn
#              2. Should we be splitting and separately batch norming the recon slice and the generated slice? It wouldn't increase
#                  computational complexity too much, I don't think, though it depends on what layer we split the bnorm in--does this split
#                  persist at all intermediate layers, or just on the output? How exactly does this split work anyhow?

# Consider replacing inception modules with inception-style context aggregation modules using dilated convolutions, or concatenating them
# to get multiple feature maps? i.e. including dilated convs as one of the possible pieces of an inception module

# Consider the information path from the intermediate layers to the hierarchical latents--can we reduce the number of parameters
# And improve expressiveness by replacing FC layers with convolutional/inception-style/context aggregation layers, and if so
# what exactly are we doing by doing this? How does replacing the FC layer with a set of conv layers differ from just FC'ing to a 
# deeper layer? Can we do this in the output of the hierarchical layer as well, replacing it with an inception upscale layer?

# Consider replacing nesterov Momentum with Adam or adamax, using our own implementation that doesn't toss preexisting grads
# Consider doing everything as SGD in the middle, then applying ADAM or Nesterov Momentum at the very end to the entire Updates dict
# (i.e. producing an "apply_adam" method a la lasagne's implementation of Nesterov Momentum)

# Consider adding in provisions to completely abandon the pixel-wise reconstruction, and only use introspection+adversarial loss
# Consider returning more meaningful adversarial loss metrics so that we can observe adversarial performance more accurately.
# Consider weighting adversarial loss so that the discriminator doesn't just learn to be dumb and let the generator walk all over it,
# i.e. running into a super-local optimum such that it always outputs "True," thereby always being right when examining a real image and
# letting the generator always get its full win when examining a generated image.
# learn the difference between "i.e." and "e.g."


# Some sort of attention-esque mechanism...maybe give each latent a particular receptive field? Or weight pixel-wise reconstruction
# by each pixel's distance from center, forcing it to produce outer details better
# Is the data shuffled? How is it ordered? Do we need to shuffle it ourselves?

# Is a 500-dim hierarchical latent representation realy equivalent to a 500-dim non-hierarchical rep if 
# it necessarily increases the number of layer parameters?

# Add in text to our generated images, indicating configuration parameters, epoch number, and accuracy, for easy reference

# Scale the adversarial gradients by the reconstruction loss so that one objective does not overtake the other


# Develop gui that allows you to explore 3D latent space
# Maybe even do that for hierarchical latent space where the lower-level inputs are set to zero. We can run inference on 32*32*32 in real time,
# we can damn well do it in 3*(2*32)*(2*32) = 12*32*32
# Version Notes:
#
# 
# introspection notes:
# Get more improvement/size of latent space; seems to also generalize better (the avg training error is closer to the validation error)

# Figure out a way to shuffle back and forth such that we reconstruct/adversarialize on alternating halves of the dataset and
# Add option to choose between nesterov momentum and adam

# Consider adding in log_sigma parameters for the introspective losses as well

# 

## Hierarchical Adversarial Autoencoder
#
# Consider throwing in a "output pictures and save only if validation accuracy improves"
# Version 3+: Changing from (0,1) to (-1,1) and using Tanh in place of sigmoids



## Notes: orthogonal regularization
# -Only applied to weight vectors! make sure this only happens on convolution filters or square matrices!
# instead of Eye, do we need to scale by eye/num_filts?
import argparse
import imp
import time
import logging
import itertools

import numpy as np
from path import Path
import theano
import theano.tensor as T
from theano.tensor.opt import register_canonicalize
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from lasagne.layers import SliceLayer as SL

import voxnet
import CAcheckpoints
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
    
    # Classification Tensor, only used if including a supervised or class-conditional task
    y = T.TensorType('float32', [False]*2)('y')
    
    # Shared and Utility Variables
    X_shared = lasagne.utils.shared_empty(4, dtype='float32')
    y_shared = lasagne.utils.shared_empty(2, dtype='float32')
    Z_shared = lasagne.utils.shared_empty(2, dtype='float32')
    pi = np.cast[theano.config.floatX](np.pi)

    
    # Input layer
    l_in = model['l_in']
    
    # Output layer
    l_out = model['l_out']
    
    # Latent Layer
    l_latents = model['l_latents']
    
    # Means
    l_mu = model['l_mu']
    
    # Log-sigmas
    l_ls = model['l_ls']
    
    # Introspective loss layers
    l_introspect = model['l_introspect']
    
    # Adversarial Discriminator
    l_discrim = model['l_discrim']
    
    # Classifier
    l_classifier = model['l_classifier']
    
    # Class-conditional latents
    l_cc = model['l_cc']
    
    # Decoder Layers, including final output layer. Consider calling this something difference to indicate that it is
    # actually a list of layers, not just a single layer.
    l_decoder = lasagne.layers.get_all_layers(l_out)[len(lasagne.layers.get_all_layers(l_latents)):]
    

    # Batch Indexing Parameters
    batch_index = T.iscalar('batch_index')
    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    
    # Define RNG
    rng = RandomStreams(lasagne.random.get_rng().randint(1,69))
   
    ###############################################################################
    # Step 1: Compute full forward pass, save the outputs of all relevant layers? #
    ###############################################################################
    
    # Build the main computational graph
    # if cfg['reconstruct'] or cfg['introspect']:
        # if cfg['adversarial']:
            # outputs = lasagne.layers.get_output([l_out]+[l_mu]+[l_ls]+\
                                                # [l_classifier]+[l_discrim]+[SL(l,indices=slice(0,cfg['batch_size']//2),axis=0)\
                                                # for l in l_introspect],\
                                                # {l_in:X, 
                                                # model['l_cc']:y,
                                                # model['l_Z_rand']:rng.normal(cfg['batch_size']//2,cfg['num_latents'])}) # Consider swapping l_classifier in for l_latents. may need to tab this properly
        # else:
        # outputs = lasagne.layers.get_output([l_out]+[lasagne.layers.ConcatLayer([lasagne.layers.flatten(mu) for mu in l_mu])]+[lasagne.layers.ConcatLayer([lasagne.layers.flatten(ls) for ls in l_ls])]+\
                                                # [l_classifier]+[l_discrim]+l_introspect,\
                                                # {l_in:X, 
                                                # model['l_cc']:y}) # Consider swapping l_classifier in for l_latents. may need to tab this properly
    # elif cfg['adversarial']:
    outputs = lasagne.layers.get_output([l_out]+[l_mu]+[l_ls]+\
                                                [l_classifier]+[l_discrim]+l_introspect,\
                                                {l_in:X, 
                                                model['l_cc']:y,
                                                model['l_Z_rand']:Z})

    # Reconstruction
    X_hat = outputs[0]
    
    # Latent means
    Z_mu = outputs[1]
    
    # Latent log-sigmas
    Z_ls = outputs[2]
    
    # Classification Predictions
    y_hat = outputs[3]
    
    # Discriminator Output
    p_X = outputs[4]
    
    # Output of the encoder layers (selected for introspection) as a function of the input image
    g_X = outputs[5:]
    
    # Build the second half of the computational graph
    # if cfg['adversarial']:
        # out_hat = lasagne.layers.get_output([l_discrim]+[SL(l,indices=slice(0,cfg['batch_size']//2),axis=0) for l in l_introspect],{l_in:X_hat})
    # else:
    # out_hat = lasagne.layers.get_output([l_discrim]+l_introspect,{l_in:X_hat})
    out_hat = lasagne.layers.get_output([l_discrim]+l_introspect,{l_in:X_hat})
    # Discriminator Output given Reconstruction/samples
    p_X_hat = out_hat[0]
    
    # Output of the encoder layers (selected for introspection) as a function of the reconstruction
    g_X_hat = out_hat[1:]
    
    p_X_gen = lasagne.layers.get_output(l_discrim,{l_in:lasagne.layers.get_output(l_out,{model['l_latents']:Z})})
    
    # Build the testing computational graph
    out_d = lasagne.layers.get_output([l_out,l_classifier]+[SL(lat,indices=slice(0,cfg['batch_size']//2),axis=0) for lat in l_latents],
                                                     {l_in:X, 
                                                      model['l_cc']:y,
                                                      model['l_Z_rand']:rng.normal((cfg['batch_size']//2,cfg['num_latents']))},
                                                     deterministic=True) if cfg['adversarial'] and cfg['reconstruct'] else\
                           lasagne.layers.get_output([l_out,                                                      
                                                      l_classifier]+l_latents,
                                                     {l_in:X, 
                                                      model['l_cc']:y},
                                                     deterministic=True) if cfg['hierarchical'] else\
                           lasagne.layers.get_output([l_out,                                                      
                                                      l_classifier,l_latents],
                                                     {model['l_Z_rand']:rng.normal((cfg['batch_size'],cfg['num_latents']))},
                                                     deterministic=True)                         
    X_hat_deterministic = out_d[0]
    y_hat_deterministic = out_d[1]
    latent_values = out_d[2:] if cfg['hierarchical'] else None
    
    #################################
    # Step 2: Define loss functions #
    #################################
    
    # Orthogonal normalization for all parameters
    # Define orthonormal residual
    def ortho_res(z):
        s = 0
        for x in z:
            if x.name[4:8] is 'conv':
                y = T.batched_tensordot(x,x.dimshuffle(0,1,3,2),[[1,3],[1,2]])
                y-=T.eye(x.shape[2],x.shape[3]).dimshuffle('x',0,1).repeat(x.shape[0],0)
                s+=T.sum(T.abs(y))
        return(s)    
        # return sum([T.sum( T.abs_(T.batched_tensordot(x,x.dimshuffle(0,1,3,2),[[1,3],[1,2]])-T.eye(x.shape[2],x.shape[3]).dimshuffle('x',0,1).repeat(x.shape[0],0))) for x in z])
        
        # return T.sum(T.batched_dot(x,x.dimshuffle(1,0,2))-T.identity_like(x))
       
         # y = T.batched_dot(x,x.T)
         # return T.sum(y-T.identity_like(x))
        # y = [T.batched_dot(x,x.T) for x in z]
        # return T.sum([T.sum(x-T.identity_like(x)) for x in y])
    l2_all = lasagne.regularization.regularize_network_params(l_out,
            lasagne.regularization.l2,tags={'regularizable':True,'trainable':True})
            
    # Create log_sigma parameter
    log_sigma_theta = lasagne.utils.create_param(spec=np.zeros((3, 64, 64)),shape=(3,64,64), name='log_sigma_theta')
    log_sigma = log_sigma_theta.dimshuffle('x', 0, 1, 2)
    
    # Define Pixel-wise reconstruction loss
    # if cfg['reconstruct'] or cfg['introspect']:
        # pixel_loss = 0.5 * T.mean(T.log(2 * pi) + 2 * log_sigma + T.sqr(lasagne.nonlinearities.sigmoid(X_hat[:cfg['batch_size']//2,:,:,:])\
                     # - X[:cfg['batch_size']//2,:,:,:]) / T.exp(2 * log_sigma)) if cfg['adversarial']\
                     # else 0.5 * T.mean(T.log(2 * pi) + 2 * log_sigma + T.sqr(lasagne.nonlinearities.sigmoid(X_hat) - X) / (T.exp(2 * log_sigma)*(T.abs_(X-0.5)+0.5)))
    # else:
        # pixel_loss = None
    # pixel_loss = 0.5 * T.mean(T.log(2 * pi) + 2 * log_sigma + T.sqr(X_hat - X) / (T.exp(2 * log_sigma)))
    # pixel_loss = T.mean(2.0*T.log(0.5*( T.exp(100*(X_hat-X)) + T.exp(-100*(X_hat-X)) ) ) )
    pixel_loss = T.mean(2*T.abs_(X_hat-X+1e-8))
    # KL Divergence between latents and Standard Normal prior
    kl_div = -0.5 * T.mean(1 + 2*Z_ls - T.sqr(Z_mu) - T.exp(2 * Z_ls))
    # kl_div = -0.5 * T.mean(1 + 2*Z_ls[:cfg['batch_size']//2,:] - T.sqr(Z_mu[:cfg['batch_size']//2,:]) - T.exp(2 * Z_ls[:cfg['batch_size']//2,:])) if cfg['adversarial'] and cfg['reconstruct']\
        # else -0.5 * T.mean(1 + 2*Z_ls - T.sqr(Z_mu) - T.exp(2 * Z_ls)) if cfg['reconstruct'] or cfg['introspect']\
        # else None
    
    # Classification objective losses
    if cfg['discriminative']:
        print('Calculating Classification Loss and Grads...')
        # Calculate Classifier Loss
        classifier_loss = T.cast(T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(y_hat), y)), 'float32')
        
        # Classifier Training Accuracy
        classifier_error_rate = T.cast( T.mean( T.neq(T.argmax(y_hat,axis=1), T.argmax(y,axis=1)) ), 'float32' )
        
        # Classifier Validation/Test Accuracy
        classifier_test_error_rate = T.cast( T.mean( T.neq(T.argmax(y_hat_deterministic,axis=1), T.argmax(y,axis=1))), 'float32' )
        
        # Combined losses
        reg_pixel_loss = pixel_loss + cfg['reg']*l2_all +classifier_loss+kl_div if cfg['kl_div'] else pixel_loss + cfg['reg']*l2_all +classifier_loss
   
    else:
        classifier_loss = None
        classifier_error_rate = None
        classifier_test_error_rate = None
        reg_pixel_loss = pixel_loss + cfg['reg']*l2_all+kl_div if cfg['kl_div'] and cfg['reconstruct'] else pixel_loss + cfg['reg']*l2_all if cfg['reconstruct'] else None
    
    ##########################
    # Step 3: Define Updates #
    ##########################
    
    # Get Parameters
    
    # All network parameters, including log_sigma
    params = lasagne.layers.get_all_params(l_out,trainable=True)+[log_sigma_theta]
    
    # Encoder Parameters
    encoder_params = lasagne.layers.get_all_params(l_latents,trainable=True)+l_discrim.get_params(trainable=True)
    
    # Decoder Params--consider including log_sigma_theta in this, too?
    decoder_params = [p for p in lasagne.layers.get_all_params(l_out,trainable=True) if p not in encoder_params]
    # decoder_params = lasagne.layers.get_all_params(l_out,trainable=True)
    Z_params = [p for p in lasagne.layers.get_all_params(l_latents,trainable=True) if p not in lasagne.layers.get_all_params(l_discrim,trainable=True)]
    
    
    # Define learning rate, with provisions made for annealing schedule
    if isinstance(cfg['learning_rate'], dict):
        learning_rate = theano.shared(np.float32(cfg['learning_rate'][0]))
    else:
        learning_rate = theano.shared(np.float32(cfg['learning_rate']))
 
   # Prepare the pixel-wise reconstruction updates
    # if cfg['reconstruct']:
        # print('Calculating Pixel-wise Loss and Grads...')
        # updates = lasagne.updates.adam(reg_pixel_loss,params,learning_rate)
    # elif cfg['kl_div']:
        # updates = lasagne.updates.adam(kl_div,encoder_params,learning_rate)
    # else:
        # updates = OrderedDict()
        # updates=lasagne.updates.adam(
    
    # grads = T.grad(cost = reg_pixel_loss, wrt = params)  # Optionally calculate gradients directly
    
    
    # Adversarial Stuff
    if cfg['adversarial']:
    
    
        print('Calculating Adversarial Loss and Grads...')
        # Regularizations
        # l2_discrim = lasagne.regularization.regularize_network_params(l_discrim,
            # lasagne.regularization.l2,tags={'regularizable':True,'trainable':True})
        l2_discrim = lasagne.regularization.apply_penalty(encoder_params,lasagne.regularization.l2)
        l2_gen = lasagne.regularization.apply_penalty(decoder_params,lasagne.regularization.l2)
        # l2_gen = lasagne.regularization.regularize_network_params(l_out,
            # lasagne.regularization.l2,tags={'regularizable':True,'trainable':True})
        
        
        # Adversarial Loss for Discriminator
        # adversarial_discrim_loss = T.mean(T.nnet.binary_crossentropy(T.clip( p_X_hat , 1e-7, 1.0 - 1e-7), T.zeros(p_X_hat.shape)))\
                                 # + T.mean(T.nnet.binary_crossentropy(T.clip( p_X , 1e-7, 1.0 - 1e-7), T.ones(p_X.shape)))+cfg['reg']*l2_discrim
        feature_loss = T.cast(T.mean([T.mean(lasagne.objectives.squared_error(g_X[i],g_X_hat[i])) for i in xrange(len(g_X_hat))]),'float32')                         
        discrim_g_loss =  T.mean(T.nnet.binary_crossentropy(T.clip( p_X_hat , 1e-7, 1.0 - 1e-7), T.zeros(p_X_hat.shape)))+\
                          T.mean(T.nnet.binary_crossentropy(T.clip( p_X_gen , 1e-7, 1.0 - 1e-7), T.zeros(p_X_gen.shape)))  
        discrim_d_loss = T.mean(T.nnet.binary_crossentropy(T.clip( p_X , 1e-7, 1.0 - 1e-7), T.ones(p_X.shape)))
        adversarial_discrim_loss = discrim_g_loss+discrim_d_loss+cfg['reg']*l2_discrim#+kl_div+cfg['recon_weight']*pixel_loss
                         
        
        # Discriminator Accuracy
        discrim_accuracy = (T.mean(T.ge(p_X,0.5))+T.mean(T.lt(p_X_hat,0.5)))/2
        
        # Adversarial Loss for Generator
        adversarial_gen_loss = T.mean(T.nnet.binary_crossentropy(T.clip( p_X_hat , 1e-7, 1.0 - 1e-7), T.ones(p_X_hat.shape)))+\
                               T.mean(T.nnet.binary_crossentropy(T.clip( p_X_gen , 1e-7, 1.0 - 1e-7), T.ones(p_X_gen.shape)))+\
                               cfg['reg']*l2_gen
        # adversarial_gen_loss = T.mean(-T.log(T.clip( p_X_hat , 1e-7, 1.0 - 1e-7)/(1-T.clip( p_X_hat , 1e-7, 1.0 - 1e-7)))) + cfg['reg']*l2_gen
        # Total Adversarial Loss
        # adversarial_loss = adversarial_discrim_loss+adversarial_gen_loss
        
        # Optional: Expressions not to backpropagate through, for use with "consider_constant" in T.grad
        # block = list(itertools.chain.from_iterable([i.get_params() for i in lasagne.layers.get_all_layers(model['l_dec_fc1'])[len(lasagne.layers.get_all_layers(model['l_latents']))+1:-1]]))
        
        # Adversarial Gradients for Discriminator
        # adversarial_discrim_grads = T.grad(cost = adversarial_discrim_loss, wrt = lasagne.layers.get_all_params(l_discrim,trainable=True))#, consider_constant = [X_hat])
        
        # Adversarial Gradients for Generator
        # adversarial_gen_grads = T.grad(cost = adversarial_gen_loss, wrt = lasagne.layers.get_all_params(l_out,trainable=True))
        
        # Prepare Adversarial Updates with Adam
        # updates = lasagne.updates.adam(adversarial_discrim_grads+adversarial_gen_grads,
                                                    # lasagne.layers.get_all_params(l_discrim,trainable=True)+decoder_params,learning_rate,beta1=cfg['beta1']) if cfg['optimizer']=='Adam'\
            # else lasagne.updates.nesterov_momentum(adversarial_discrim_grads+adversarial_gen_grads,
                                                    # lasagne.layers.get_all_params(l_discrim,trainable=True)+decoder_params,learning_rate,cfg['momentum'])
        discrim_updates = lasagne.updates.adam(adversarial_discrim_loss,encoder_params,learning_rate,beta1=cfg['beta1'])
        discrim_to_latent_updates = lasagne.updates.adam(cfg['feature_weight']*feature_loss+cfg['recon_weight']*pixel_loss+kl_div,
                                                        Z_params,
                                                        learning_rate=learning_rate,beta1=cfg['beta1'])
        for ud in discrim_to_latent_updates:
            discrim_updates[ud] = discrim_to_latent_updates[ud]
        
        gen_updates = lasagne.updates.adam(adversarial_gen_loss+cfg['recon_weight']*pixel_loss+cfg['feature_weight']*feature_loss,decoder_params,learning_rate,beta1=cfg['beta1'])
        
        # for param in lasagne.layers.get_all_params(model['l_mu'],trainable=True)[-3:]+lasagne.layers.get_all_params(model['l_ls'],trainable=True)[-3:]:
            # gen_updates[param] = lasagne.updates.adam(cfg['feature_weight']*feature_loss+cfg['recon_weight']*pixel_loss+kl_div,[param],learning_rate,beta1=cfg['beta1'])
        for ud in discrim_to_latent_updates:
            gen_updates[ud] = discrim_to_latent_updates[ud]
        # if cfg['reconstruct'] or cfg[:
        # for param in adversarial_updates:
            # updates[param] = updates[param] + adversarial_updates[param] - param if param in updates else adversarial_updates[param]
        # Prepare Adversarial Updates with Nesterov Momentum
        # for param,grad in zip(lasagne.layers.get_all_params(l_discrim,trainable=True)+decoder_params,adversarial_discrim_grads+adversarial_gen_grads):
            # value = param.get_value(borrow=True)
            # velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 # broadcastable=param.broadcastable)
            # x = cfg['momentum'] * velocity - learning_rate*grad
            # updates[velocity] = x
            # updates[param] = updates[param] + x*cfg['momentum'] - learning_rate*grad if param in updates else param + x*cfg['momentum'] - learning_rate*grad
    else:
        adversarial_gen_loss = None
        adversarial_discrim_loss = None
    
    if cfg['introspect']:
        print('Calculating Introspective Loss and Grads...')
        
        # Introspective Loss Term
        # Optionally include term to scale losses such that deeper layers are considered more important than more shallow layers
        feature_loss = T.cast(T.mean([T.mean(lasagne.objectives.squared_error(g_X[i],g_X_hat[i])) for i in xrange(len(g_X_hat))]),'float32')
        
        # Introspective Gradients
        feature_grads = lasagne.updates.get_or_compute_grads(feature_loss,decoder_params)
        
        # Prepare Introspective Updates with Adam
        feature_updates = lasagne.updates.adam(feature_grads,decoder_params,learning_rate) if cfg['optimizer']=='Adam'\
            else lasagne.updates.nesterov_momentum(feature_grads,decoder_params,learning_rate,cfg['momentum'])
            
        for param in feature_updates:
            updates[param] = updates[param] + feature_updates[param] - param if param in updates else feature_updates[param]
        
        # Prepare Introspective Updates with Nesterov Momentum
        # for param,grad in zip(decoder_params,feature_grads):
            # value = param.get_value(borrow=True)
            # velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 # broadcastable=param.broadcastable)
            # x = cfg['momentum'] * velocity - learning_rate*grad
            # updates[velocity] = x
            # updates[param] = updates[param] + x*cfg['momentum'] - learning_rate*grad if param in updates else param + x*cfg['momentum'] - learning_rate * grad

         
    else:
        feature_loss = None
    
    # Pixel-wise Training MSE
    # error_rate = T.cast( T.mean( T.sqr(lasagne.nonlinearities.sigmoid(X_hat[:cfg['batch_size']//2,:,:,:])-X[:cfg['batch_size']//2,:,:,:])), 'float32' ) if cfg['adversarial'] and cfg['reconstruct']\
        # else T.cast( T.mean( T.sqr(lasagne.nonlinearities.sigmoid(X_hat)-X)), 'float32' ) if cfg['reconstruct'] or cfg['introspect']\
        # else None
    error_rate = T.cast( T.mean( T.sqr(X_hat-X)), 'float32' )
    # Pixel-wise Test MSE
    test_error_rate = T.cast( T.mean( T.sqr(lasagne.nonlinearities.sigmoid(X_hat_deterministic[:cfg['batch_size']//2,:,:,:])-X[:cfg['batch_size']//2,:,:,:])), 'float32' ) if cfg['adversarial'] and cfg['reconstruct']\
        else T.cast( T.mean( T.sqr(lasagne.nonlinearities.sigmoid(X_hat_deterministic)-X)), 'float32' ) if cfg['reconstruct'] or cfg['introspect']\
        else None
      
    # Sample function
    if cfg['hierarchical']:
        sample = theano.function([Z],lasagne.nonlinearities.sigmoid(lasagne.layers.get_output(model['l_out'],\
                {l_Z:T.reshape(Z[:,a[0]:a[1]],(T.shape(Z)[0],)+dim) for l_Z,a,dim in zip(model['l_latents'], zip(np.append(0,cfg['latent_indices'][:-1]),cfg['latent_indices']),cfg['latent_dims'])},deterministic=True)),on_unused_input='warn')
    else:
        sample = theano.function([Z],lasagne.layers.get_output(model['l_out'],{model['l_latents']:Z},deterministic=True))
    # Inference Function--Infer latents given an image
    # Zfn = theano.function([X],T.concatenate([T.flatten(l,2) for l in latent_values],axis=1),on_unused_input='warn') if cfg['reconstruct'] or cfg['introspect'] else None
    Zfn = theano.function([X],lasagne.layers.get_output(model['l_latents'],{model['l_in']:X},deterministic=True),on_unused_input='warn')
    
  
    # Outputs for Update Function
    update_outs = [x for x in [pixel_loss, 
            feature_loss,
            classifier_loss,
            adversarial_gen_loss,
            adversarial_discrim_loss,
            kl_div,
            classifier_error_rate,
            error_rate,
            ] if x is not None]
    
    # Define Update Function
    # update_iter = theano.function([batch_index],update_outs,
            # updates=updates, givens={
            # X: X_shared[batch_slice],
            # y: y_shared[batch_slice],
            # Z: Z_shared[batch_slice],
        # },on_unused_input='warn' )
    
    
    update_gen = theano.function([batch_index],[adversarial_gen_loss,pixel_loss,1-error_rate],
                                 updates=gen_updates,
                                 givens = {X: X_shared[batch_slice], y: y_shared[batch_slice],Z: Z_shared[batch_slice]},
                                 on_unused_input = 'warn')
                                 
    update_discrim = theano.function([batch_index],[discrim_g_loss,discrim_d_loss,discrim_accuracy,pixel_loss,1-error_rate],
                                 updates=discrim_updates,
                                 givens = {X: X_shared[batch_slice], y: y_shared[batch_slice],Z: Z_shared[batch_slice]},
                                 on_unused_input = 'warn')                             
    # outputs for test/validation function
    test_outs =  [x for x in [test_error_rate,
                        classifier_test_error_rate] if x is not None]   
    
    # Define test/validation function
    test_error_fn = theano.function([batch_index], 
            test_outs, givens={
            X: X_shared[batch_slice],
            y: y_shared[batch_slice]           
        },on_unused_input='warn' )

    # Dictionary of Theano Functions
    # tfuncs = {'update_iter':update_iter,
    tfuncs = {'update_gen': update_gen,
             'update_discrim': update_discrim,
             'test_function':test_error_fn,
             'sample': sample,
             'Zfn' : Zfn
            }
            
    # Dictionary of Theano Variables        
    tvars = {'X' : X,
             'y' : y,
             'Z' : Z,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'Z_shared' : Z_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index,
             'learning_rate' : learning_rate,
             'log_sigma': log_sigma_theta
            }
            
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
    
    # Define the name of the jsonl file to which the training log will be saved
    metrics_fname = weights_fname[:-4]+'METRICS.jsonl'
    
    # Prepare logs
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    logging.info('Metrics will be saved to {}'.format(metrics_fname))
    mlog = voxnet.metrics_logging.MetricsLogger(metrics_fname, reinitialize=True)
    model = config_module.get_model(interp=False)
    
    logging.info('Compiling theano functions...')
    
    # Compile functions
    tfuncs, tvars,model = make_training_functions(cfg,model)

    logging.info('Training...')
    
    # Iteration Counter, indicates total number of minibatches processed
    itr = 0
    
    # Best validation accuracy variable
    best_acc = 0
    
    # Test set for interpolations
    test_set = CelebA('64',('test',),sources=('features',))    

    # Loop across epochs
    offset = True
    params = list(set(lasagne.layers.get_all_params(model['l_out'],trainable=True)+[tvars['log_sigma']]+lasagne.layers.get_all_params(model['l_discrim'],trainable=True)+[x for x in lasagne.layers.get_all_params(model['l_out']) if x.name[-4:]=='mean' or x.name[-7:]=='inv_std']))
    
    # Ratio of gen updates to discrim updates
    update_ratio = cfg['update_ratio']
    for epoch in xrange(cfg['max_epochs']):
        offset = not offset
        
        # Get generator for data
        loader = data_loader(cfg,
                            CelebA('64',('train',),sources=('features',)),
                            offset=offset*cfg['batch_size']//2,shuffle=cfg['shuffle'],
                            seed=epoch) # Does this need to happen every epoch?
        
        # Update Learning Rate, either with annealing schedule or decay rate
        if isinstance(cfg['learning_rate'], dict) and epoch > 0:
            if any(x==epoch for x in cfg['learning_rate'].keys()):
                lr = np.float32(tvars['learning_rate'].get_value())
                new_lr = cfg['learning_rate'][epoch]
                logging.info('Changing learning rate from {} to {}'.format(lr, new_lr))
                tvars['learning_rate'].set_value(np.float32(new_lr))
        if cfg['decay_rate'] and epoch > 0:
            lr = np.float32(tvars['learning_rate'].get_value())
            new_lr = lr*(1-cfg['decay_rate'])
            logging.info('Changing learning rate from {} to {}'.format(lr, new_lr))
            tvars['learning_rate'].set_value(np.float32(new_lr))
        
        # Number of Chunks
        iter_counter = 0
        
        # Epoch-Wise Metrics
        # vloss_e, floss_e, closs_e, a_g_loss_e, a_d_loss_e, d_kl_e, c_acc_e, acc_e = 0, 0, 0, 0, 0, 0, 0, 0   

        # Loop across all chunks   
        for x_shared in loader:
     
            # Increment Chunk Counter
            iter_counter+=1
            
            # Figure out number of batches
            num_batches = len(x_shared)//cfg['batch_size']
            
            # Shuffle chunk
            # np.random.seed(42*epoch)
            index = np.random.permutation(len(x_shared))
            
            # Load data onto GPU
            tvars['X_shared'].set_value(x_shared[index], borrow=True)
            tvars['Z_shared'].set_value(np.float32(np.random.randn(len(x_shared),cfg['num_latents'])),borrow=True)
            
            # Chunk Metrics
            # voxel_lvs,feature_lvs,class_lvs,a_g_lvs,a_d_lvs, kl_divs,class_accs,accs = [],[],[],[],[],[],[],[]            
            a_g_lvs,a_dg_lvs,a_dd_lvs,discrim_acc,pixel_lvs,pixel_accs = [],[],[],[],[],[]
            # Loop across all batches in chunk
            for bi in xrange(num_batches):
                
                if itr % (update_ratio+1)==0:
                    [a_gen_loss,pixel_loss,pixel_acc] = tfuncs['update_gen'](bi)
                    a_g_lvs.append(a_gen_loss)
                    pixel_lvs.append(pixel_loss)
                    pixel_accs.append(pixel_acc)
                else:
                    [a_discrim_gloss,a_discrim_dloss,discrim_accuracy,pixel_loss,pixel_acc] = tfuncs['update_discrim'](bi)
                    a_dg_lvs.append(a_discrim_gloss)
                    a_dd_lvs.append(a_discrim_dloss)
                    discrim_acc.append(discrim_accuracy)
                    pixel_lvs.append(pixel_loss)
                    pixel_accs.append(pixel_acc)
                
                itr += 1    
                
            # if not a_dg_lvs:
                # a_dg_lvs,a_dd_lvs,discrim_acc = 0,0,0
            # if not a_g_lvs:
                # a_g_lvs = 0
                   
                # Train!
                # results = tfuncs['update_iter'](bi)
                
                # Assign results
                # TODO: Clean up the assignment so that the variable things are just on the end of the assignment and
                # this can be done in one or two lines
                # voxel_loss = results[0] if cfg['reconstruct'] or cfg['introspect'] else 0
                # feature_loss = results[(cfg['reconstruct'] or cfg['introspect'])] if cfg['introspect'] else 0 
                # classifier_loss = results[cfg['introspect']+(cfg['reconstruct'] or cfg['introspect'])] if cfg['discriminative'] else 0
                # a_gen_loss = results[cfg['introspect']+cfg['discriminative']+(cfg['reconstruct'] or cfg['introspect'])] if cfg['adversarial'] else 0
                # a_discrim_loss = results[1+cfg['introspect']+cfg['discriminative']+(cfg['reconstruct'] or cfg['introspect'])] if cfg['adversarial'] else 0
                # kl_div = results[cfg['introspect']+cfg['discriminative']+2*cfg['adversarial']+(cfg['reconstruct'] or cfg['introspect'])] if cfg['kl_div'] else 0
                # class_acc = results[cfg['introspect']+cfg['discriminative']+2*cfg['adversarial']+cfg['kl_div']+(cfg['reconstruct'] or cfg['introspect'])] if cfg['discriminative'] else 0
                # acc = results[cfg['introspect']+2*cfg['discriminative']+2*cfg['adversarial']+cfg['kl_div']+(cfg['reconstruct'] or cfg['introspect'])] if cfg['reconstruct'] or cfg['introspect'] else 0              
                # voxel_lvs.append(voxel_loss)
                # feature_lvs.append(feature_loss)
                # class_lvs.append(classifier_loss)
                
                
                # kl_divs.append(kl_div)
                # class_accs.append(class_acc)
                # accs.append(acc)

                
            [agloss,adgloss,addloss,accuracy] = [float(np.mean(a_g_lvs)),float(np.mean(a_dg_lvs)),float(np.mean(a_dd_lvs)),float(np.mean(discrim_acc))]
            [ploss,pixel_accuracy] = [float(np.mean(pixel_lvs)),float(np.mean(pixel_accs))]            
            # Chunk-wise metrics
            # [vloss, floss,closs, agloss, adloss, d_kl,c_acc,acc] = [float(np.mean(voxel_lvs)), float(np.mean(feature_lvs)),
                                                    # float(np.mean(class_lvs)), float(np.mean(a_g_lvs)),float(np.mean(a_d_lvs)),float(np.mean(kl_divs)),
                                                    # 1.0-float(np.mean(class_accs)), 1.0-float(np.mean(accs))]
            # Epoch-wise metrics                                                    
            # vloss_e, floss_e, closs_e, a_g_loss_e, a_d_loss_e, d_kl_e, c_acc_e, acc_e = [vloss_e+vloss, floss_e+floss, closs_e+closs, a_g_loss_e+agloss, a_d_loss_e+adloss, d_kl_e+d_kl, c_acc_e+c_acc, acc_e+acc] 
            
            # Report Chunk Metrics
            # logging.info('epoch: {:4d}, itr: {:8d}, p_loss: {:8.5f}, f_loss: {:8.5f}, a_g_loss: {:8.5f}, a_d_loss: {:8.5f}, D_kl: {:8.5f}, acc: {:6.5f}'.format(epoch, itr, vloss, floss,
                                                                                                                           # agloss, adloss, d_kl, acc))
            logging.info('epoch: {:4d}, itr: {:8d}, ag_loss: {:7.4f}, adg_loss: {:7.4f}, add_loss: {:7.4f}, acc: {:5.3f}, ploss: {:7.4f}, pacc: {:5.3f}'.format(epoch,itr,agloss,adgloss,addloss,accuracy,ploss,pixel_accuracy))
            mlog.log(epoch=epoch, itr=itr, agloss=agloss,adgloss = adgloss,addloss=addloss,discrim_accuracy=accuracy,ploss=ploss,pixel_accuracy=pixel_accuracy)
            # Log Chunk Metrics
            # mlog.log(epoch=epoch, itr=itr, vloss=vloss,floss=floss, agloss=agloss,adloss = adloss, acc=acc,d_kl=d_kl,c_acc=c_acc)
            
        # Average Epoch-wise Metrics
        # vloss_e, floss_e, closs_e, a_g_loss_e, a_d_loss_e, d_kl_e, c_acc_e, acc_e = [vloss_e/iter_counter, floss_e/iter_counter, 
                                                             # closs_e/iter_counter, a_g_loss_e/iter_counter, a_d_loss_e/iter_counter, d_kl_e/iter_counter,
                                                             # c_acc_e/iter_counter, acc_e/iter_counter]
        # Report Epoch-wise metrics
        # logging.info('Training metrics, Epoch {}, p_loss: {}, f_loss: {}, a_g_loss: {}, a_d_loss: {}, c_loss: {}, D_kl: {}, class_acc: {}, acc: {}'.format(epoch, vloss_e, floss_e,a_g_loss_e, a_d_loss_e, closs_e,d_kl_e,c_acc_e,acc_e))
        
        # Log Epoch-wise metrics
        # mlog.log(epoch=epoch, vloss_e=vloss_e, floss_e=floss_e, a_g_loss_e = a_g_loss_e,a_d_loss_e=a_d_loss_e,closs_e=closs_e, d_kl_e=d_kl_e, c_acc_e=c_acc_e, acc_e=acc_e)
        

        
        if cfg['reconstruct'] or cfg['introspect']:
            logging.info('Examining performance on validation set')
            
            # Validation Metrics
            test_error,test_class_error = [],[],
            
            # Prepare Test Loader
            for o in xrange(2):
                test_loader = data_loader(cfg,CelebA('64',('valid',),sources=('features',)),offset=o*cfg['batch_size']//2)
                
                # Loop Across Chunks
                for x_shared in test_loader:
                
                    # Figure Out Number of Batches
                    num_batches = len(x_shared)//cfg['batch_size']
                    
                    # Load chunk onto GPU
                    tvars['X_shared'].set_value(x_shared, borrow=True)

                    # Loop Across Batches
                    for bi in xrange(num_batches):
                    
                        # Test!
                        test_results = tfuncs['test_function'](bi) # Get the test 
                        
                        # Assign results
                        batch_test_error=test_results[0]
                        batch_test_class_error = test_results[1] if cfg['discriminative'] else 0
                        test_error.append(batch_test_error)
                        test_class_error.append(batch_test_class_error)
                        

                    
            # Average Results        
            t_error = 1-float(np.mean(test_error))
            t_class_error = 1-float(np.mean(test_class_error))        
            
            # Report Validation Results
            logging.info('Epoch {} Test Accuracy: {}, Classification Test Accuracy: {}, Best_acc: {} '.format(epoch, t_error,t_class_error,best_acc))
            
            # Log Validation Results
            mlog.log(test_error=t_error,t_class_error = t_class_error)

               
        # If we see improvement, save weights and produce output images
        # if cfg['reconstruct'] or cfg['introspect']:
        if not (epoch%cfg['checkpoint_every_nth']):
            
        
            # Update Best-yet accuracy
            # if t_error > best_acc:
                # best_acc = t_error 
                
                # Save Weights
                                               
        
            # Open Test Set
            test_set.open()
            
            np.random.seed(epoch*42+5)
            # Generate Random Samples
            samples = np.uint8(from_tanh(tfuncs['sample'](np.random.randn(27,cfg['num_latents']).astype(np.float32))))
            
            
            np.random.seed(epoch*42+5)
            # Get Reconstruction/Interpolation Endpoints
            endpoints = np.uint8(test_set.get_data(request = list(np.random.choice(test_set.num_examples,6,replace=False)))[0])
           
            # Get reconstruction latents
            Ze = np.asarray(tfuncs['Zfn'](to_tanh(np.float32(endpoints))))
          
            print(np.shape(Ze))               
            
            # Get Interpolant Latents
            Z = np.asarray([Ze[2 * i, :] * (1 - j) + Ze[2 * i + 1, :] * j  for i in range(3) for j in [x/6.0 for x in range(7)]],dtype=np.float32)
            
            # Get all images
            images = np.append(samples,np.concatenate([np.insert(endpoints[2*i:2*(i+1),:,:,:],1,np.uint8(from_tanh(tfuncs['sample'](Z[7*i:7*(i+1),:]))),axis=0) for i in range(3)],axis=0),axis=0)


            # Plot images
            plot_image_grid(images,6,9,'pics/'+str(args.config_path)[:-3]+'_'+str(epoch)+'.png')
            
            # Close test set
            test_set.close(state=None)
            params = list(set(lasagne.layers.get_all_params(model['l_out'],trainable=True)+[tvars['log_sigma']]+lasagne.layers.get_all_params(model['l_discrim'],trainable=True)+[x for x in lasagne.layers.get_all_params(model['l_out'])+lasagne.layers.get_all_params(model['l_discrim']) if x.name[-4:]=='mean' or x.name[-7:]=='inv_std']))
            GANcheckpoints.save_weights(weights_fname, params,{'itr': itr, 'ts': time.time(),'learning_rate':np.float32(tvars['learning_rate'].get_value())})
                # Save weights
                # CAcheckpoints.save_weights(weights_fname, model['l_out'],tvars['log_sigma'],
                                                    # {'itr': itr, 'ts': time.time(),'best_acc':best_acc,'learning_rate':np.float32(tvars['learning_rate'].get_value())}) 
        # elif not (epoch%cfg['checkpoint_every_nth']):
            # logging.info('Checkpoint: Saving weights and generating samples') 
            # np.random.seed(epoch*42+5)
            # samples = np.uint8(from_tanh(tfuncs['sample'](np.random.randn(54,cfg['num_latents']).astype(np.float32))))
            # Plot images
            # plot_image_grid(samples,6,9,'pics/'+str(args.config_path)[:-3]+'_'+str(epoch)+'.png')
            # params = list(set(lasagne.layers.get_all_params(model['l_out'],trainable=True)+[tvars['log_sigma']]+lasagne.layers.get_all_params(model['l_discrim'],trainable=True)+[x for x in lasagne.layers.get_all_params(model['l_out'])+lasagne.layers.get_all_params(model['l_discrim']) if x.name[-4:]=='mean' or x.name[-7:]=='inv_std']))
            # GANcheckpoints.save_weights(weights_fname, params,{'itr': itr, 'ts': time.time(),'learning_rate':np.float32(tvars['learning_rate'].get_value())})            
    logging.info('training done')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='config .py file')
    args = parser.parse_args()
    main(args)
