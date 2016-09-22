### Introspective Adversarial Network Training Function
# A Brock, 2016

import argparse
import imp
import time
import logging
import itertools
import os
import string

import numpy as np
from path import Path
import theano
import theano.tensor as T
from theano.tensor.opt import register_canonicalize
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from lasagne.layers import SliceLayer as SL

import metrics_logging
import GANcheckpoints
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from fuel.datasets import CelebA
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
    
    # Ternary classification values
    p1,p2,p3 = T.TensorType('int32',[False]*2)('p1'),T.TensorType('int32',[False]*2)('p2'),T.TensorType('int32',[False]*2)('p3')
    
    # Shared and Utility Variables
    X_shared = lasagne.utils.shared_empty(4, dtype='float32')
    y_shared = lasagne.utils.shared_empty(2, dtype='float32')
    Z_shared = lasagne.utils.shared_empty(2, dtype='float32')
    p1_shared = lasagne.utils.shared_empty(2, dtype='int32')
    p2_shared = lasagne.utils.shared_empty(2, dtype='int32')
    p3_shared = lasagne.utils.shared_empty(2, dtype='int32')
    pi = np.cast[theano.config.floatX](np.pi)

    
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

    # Batch Indexing Parameters
    batch_index = T.iscalar('batch_index')
    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    
    # Define RNG
    rng = RandomStreams(lasagne.random.get_rng().randint(1,69))
   
    ###############################################################################
    # Step 1: Compute full forward pass, save the outputs of all relevant layers? #
    ###############################################################################
    
    # Build the main computational graph
    outputs = lasagne.layers.get_output([l_out]+[l_mu]+[l_ls]+[l_discrim]+[l_IAF_mu]+[l_IAF_ls]+l_introspect,{l_in:X})
    # outputs = lasagne.layers.get_output([l_out]+[l_mu]+[l_ls]+[l_discrim]+l_introspect,{l_in:X})
    # Reconstruction
    X_hat = outputs[0]
    
    # Latent means
    Z_mu = outputs[1]
    
    # Latent log-sigmas
    Z_ls = outputs[2]
    
    # Discriminator Output
    p_X = outputs[3]
    
    # Latent IAF mus
    Z_IAF_mu = outputs[4]
    
    # Latent IAF logsigma
    Z_IAF_ls = outputs[5]
       
    # Output of the encoder layers (selected for introspection) as a function of the input image
    g_X = outputs[6:]
    
    # Build the second half of the computational graph
    out_hat = lasagne.layers.get_output([l_discrim]+l_introspect,{l_in:X_hat})
    
    # Discriminator Output given Reconstruction
    p_X_hat = out_hat[0]
    
    # Output of the encoder layers (selected for introspection) as a function of the reconstruction
    g_X_hat = out_hat[1:]
    
    # Discriminator output given random samples
    p_X_gen = lasagne.layers.get_output(l_discrim,{l_in:lasagne.layers.get_output(l_out,{l_Z_IAF:Z})})
    
    
    #################################
    # Step 2: Define loss functions #
    #################################
    
    # Orthogonal normalization for all parameters
    # Define orthonormal residual
    def ortho_res(z):
        s = 0
        for x in z:
            if x.name[-1] is 'W' and x.ndim==4:
                y = T.batched_tensordot(x,x.dimshuffle(0,1,3,2),[[1,3],[1,2]])
                y-=T.eye(x.shape[2],x.shape[3]).dimshuffle('x',0,1).repeat(x.shape[0],0)
                s+=T.sum(T.abs_(y))
        return(s)    

    
    # Define Pixel-wise reconstruction loss
    pixel_loss = T.mean(2*T.abs_(X_hat-X+1e-8))
    
    # KL Divergence between latents and Standard Normal prior
    kl_div = -0.5 * T.mean(1 + 2*Z_ls - T.sqr(Z_mu) - T.exp(2 * Z_ls))
    # kl_div = -T.maximum(0.5, T.mean(0.5 * (1 + 2*Z_ls - T.sqr(Z_mu) - T.exp(2 * Z_ls)) + Z_IAF_ls))

    

    ##########################
    # Step 3: Define Updates #
    ##########################
    
    # Get Parameters
    
    # All network parameters, including log_sigma
    params = lasagne.layers.get_all_params(l_out,trainable=True)
    
    # Encoder Parameters
    encoder_params = lasagne.layers.get_all_params(l_discrim,trainable=True)
    
    # MADE parameters, along with a thing to prevent the IAF params from being trained    
    Z_params = [p for p in lasagne.layers.get_all_params(l_Z_IAF,trainable=True) if p not in lasagne.layers.get_all_params(l_discrim,trainable=True)]
    print(Z_params)    
    
    # Decoder Params
    decoder_params = [p for p in lasagne.layers.get_all_params(l_out,trainable=True) if p not in lasagne.layers.get_all_params(l_Z,trainable=True)]    
    
    # Define learning rate, with provisions made for annealing schedule
    if isinstance(cfg['learning_rate'], dict):
        learning_rate = theano.shared(np.float32(cfg['learning_rate'][0]))
    else:
        learning_rate = theano.shared(np.float32(cfg['learning_rate']))
 

    
    # Adversarial Stuff
    
    
    
    print('Calculating Adversarial Loss and Grads...')
    # Regularization terms
    
    l2_Z = cfg['reg']*lasagne.regularization.apply_penalty([p for p in lasagne.layers.get_all_params(l_Z_IAF,trainable=True,regularizable=True)\
                                                    if p not in lasagne.layers.get_all_params(l_discrim,trainable=True)],
                                                    lasagne.regularization.l2)
    if 'ortho' in cfg:
        print('Applying orthogonal regularization...')
        l2_discrim = cfg['ortho']*lasagne.regularization.apply_penalty(lasagne.layers.get_all_params(l_Z,trainable=True,regularizable=True)\
                                                         +l_discrim.get_params(trainable=True,regularizable=True),
                                                          ortho_res)
                                                             
        l2_gen = cfg['ortho']*lasagne.regularization.apply_penalty([p for p in lasagne.layers.get_all_params(l_out,trainable=True,regularizable=True) if p not in encoder_params],
                                                      ortho_res)
    
    
    # Adversarial Loss for Discriminator
    
    # Discriminator loss for reconstructed and generated samples  
    # print(p_X_hat.shape[0])
    discrim_g_loss = T.mean(T.nnet.categorical_crossentropy(p_X_hat,p2)) + T.mean(T.nnet.categorical_crossentropy(p_X_gen,p3))
      
    # 
                     
    
    # Discriminator loss 
    discrim_d_loss = T.mean(T.nnet.categorical_crossentropy(p_X, p1))
    
    adversarial_discrim_loss = cfg['dg_weight']*discrim_g_loss+cfg['dd_weight']*discrim_d_loss
                     
    
    # Discriminator Accuracy
    discrim_accuracy = (T.mean(T.eq(T.argmax(p_X,axis=1),T.argmax(p1,axis=1)))+T.mean(T.eq(T.argmax(p_X_hat,axis=1),T.argmax(p2,axis=1)))+T.mean(T.eq(T.argmax(p_X_gen,axis=1),T.argmax(p3,axis=1))))/3.0
    
    
    # Feature Reconstruction Loss for Generator
    feature_loss = T.cast(T.mean([T.mean(lasagne.objectives.squared_error(g_X[i],g_X_hat[i])) for i in xrange(len(g_X_hat))]),'float32')
    
    # Adversarial loss for Generator
    gen_recon_loss = T.mean(T.nnet.categorical_crossentropy(p_X_hat,p1)) 
    gen_sample_loss = T.mean(T.nnet.categorical_crossentropy(p_X_gen,p1))
                           
    adversarial_gen_loss = cfg['agr_weight']*gen_recon_loss+cfg['ags_weight']*gen_sample_loss                        
    
    # Updates for discriminator      
    discrim_updates = lasagne.updates.adam(T.grad(adversarial_discrim_loss+l2_discrim,encoder_params,consider_constant=[X_hat]),encoder_params,learning_rate,beta1=cfg['beta1'])       
    
    # Updates for Generator
    gen_updates = lasagne.updates.adam(adversarial_gen_loss+\
                                       cfg['recon_weight']*pixel_loss+\
                                       cfg['feature_weight']*feature_loss+\
                                       l2_gen,decoder_params,learning_rate,beta1=cfg['beta1'])
    
    # Optional Inference mini-network updates--only updated based on reconstructions?
    # Z_gen_updates = lasagne.updates.adam(adversarial_gen_loss+cfg['feature_weight']*feature_loss+cfg['recon_weight']*pixel_loss+kl_div,Z_params,learning_rate=learning_rate,beta1=cfg['beta1'])
    # Z_gen_updates = lasagne.updates.adam(adversarial_gen_loss+cfg['feature_weight']*feature_loss+cfg['recon_weight']*pixel_loss+kl_div,Z_params,learning_rate=learning_rate,beta1=cfg['beta1'])
    
    # Z_discrim_updates = lasagne.updates.adam(adversarial_gen_losscfg['feature_weight']*feature_loss+cfg['recon_weight']*pixel_loss+kl_div,Z_params,learning_rate=learning_rate,beta1=cfg['beta1'])                                                
    Z_gen_updates = lasagne.updates.adam(cfg['feature_weight']*feature_loss+\
                                         cfg['recon_weight']*pixel_loss+\
                                         adversarial_gen_loss+\
                                         kl_div+\
                                         l2_Z,
                                         Z_params,
                                         learning_rate=learning_rate,
                                         beta1=cfg['beta1'])
    for ud in Z_gen_updates:
        gen_updates[ud] = Z_gen_updates[ud]
        discrim_updates[ud] = Z_gen_updates[ud]

    # Pixel-Wise MSE for reporting
    error_rate = T.cast( T.mean( T.sqr(X_hat-X)), 'float32' )
 
      
    # Sample function
    sample = theano.function([Z],lasagne.layers.get_output(l_out,{l_Z_IAF:Z},deterministic=True),on_unused_input='warn')
    
    # Inference Function--Infer non-IAF_latents given an input X    
    Zfn = theano.function([X],lasagne.layers.get_output(l_Z_IAF,{l_in:X},deterministic=True),on_unused_input='warn')
    

  
    # gen dictionary
    gd = OrderedDict()
    gd['gen_recon_loss'] = gen_recon_loss
    gd['gen_sample_loss'] = gen_sample_loss
    gd['pixel_loss'] = pixel_loss
    gd['feature_loss'] = feature_loss
    gd['pixel_acc'] = 1-error_rate
                
    # discrim dictionary
    dd = OrderedDict()
    dd['discrim_g_loss'] = discrim_g_loss
    dd['discrim_d_loss'] = discrim_d_loss
    dd['discrim_acc'] = discrim_accuracy
    dd['pixel_loss'] = pixel_loss
    dd['pixel_acc'] = 1-error_rate            
   

    update_gen = theano.function([batch_index],[gd[i] for i in gd],#[adversarial_gen_loss,pixel_loss,1-error_rate],
                                 updates=gen_updates,
                                 givens = {X: X_shared[batch_slice],
                                           y: y_shared[batch_slice],
                                           Z: Z_shared[batch_slice],
                                           p1:p1_shared[batch_slice],
                                           p2:p2_shared[batch_slice],
                                           p3:p3_shared[batch_slice]},
                                 on_unused_input = 'warn')
                                 
    update_discrim = theano.function([batch_index],[dd[i] for i in dd],#[discrim_g_loss,discrim_d_loss,discrim_accuracy,pixel_loss,1-error_rate],
                                 updates=discrim_updates,
                                 givens = {X: X_shared[batch_slice],
                                           y: y_shared[batch_slice],
                                           Z: Z_shared[batch_slice],
                                           p1:p1_shared[batch_slice],
                                           p2:p2_shared[batch_slice],
                                           p3:p3_shared[batch_slice]},
                                 on_unused_input = 'warn')                             

    # Dictionary of Theano Functions
    # tfuncs = {'update_iter':update_iter,
    tfuncs = {'update_gen': update_gen,
             'update_discrim': update_discrim,
             'sample': sample,
             'Zfn' : Zfn,
            }
            
    # Dictionary of Theano Variables        
    tvars = {'X' : X,
             'y' : y,
             'Z' : Z,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'Z_shared' : Z_shared,
             'p1' : p1_shared,
             'p2' : p2_shared,
             'p3' : p3_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index,
             'learning_rate' : learning_rate,
             'gd' : gd,
             'dd': dd
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
    mlog = metrics_logging.MetricsLogger(metrics_fname, reinitialize=(not args.resume))
    model = config_module.get_model(interp=False)
    
    logging.info('Compiling theano functions...')
    
    # Compile functions
    tfuncs, tvars,model = make_training_functions(cfg,model)

    # Shuffle Initial masks
    model['l_IAF_mu'].shuffle("Once")
    model['l_IAF_ls'].shuffle("Once")
    logging.info('Training...')
    
    # Iteration Counter, indicates total number of minibatches processed
    itr = 0
    
    # Best validation accuracy variable
    best_acc = 0
    
    # Test set for interpolations
    test_set = CelebA('64',('test',),sources=('features',))    

    # Loop across epochs
    offset = True
    params = list(set(lasagne.layers.get_all_params(model['l_out'],trainable=True)+\
                              lasagne.layers.get_all_params(model['l_discrim'],trainable=True)+\
                              [x for x in lasagne.layers.get_all_params(model['l_out'])+\
                                lasagne.layers.get_all_params(model['l_discrim'])if x.name[-4:]=='mean' or x.name[-7:]=='inv_std']))
    if os.path.isfile(weights_fname) and args.resume:
        metadata = GANcheckpoints.load_weights(weights_fname, params)
        min_epoch = metadata['epoch']+1 if 'epoch' in metadata else 0
        new_lr = metadata['learning_rate'] if 'learning_rate' in metadata else cfg['lr_schedule'][0]
        tvars['learning_rate'].set_value(np.float32(new_lr))
        print('loading weights, epoch is '+str(min_epoch),'lr is '+str(new_lr)+'.')
    else:
        min_epoch = 0
        
        
    # Ratio of gen updates to discrim updates
    update_ratio = cfg['update_ratio']
    n_shuffles = 0
    for epoch in xrange(min_epoch,cfg['max_epochs']):
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
            
            # Ternary adversarial objectives
            tvars['p1'].set_value(np.asarray([[1,0,0]]*len(x_shared),dtype=np.int32))
            tvars['p2'].set_value(np.asarray([[0,1,0]]*len(x_shared),dtype=np.int32))
            tvars['p3'].set_value(np.asarray([[0,0,1]]*len(x_shared),dtype=np.int32))
            # Chunk Metrics
            metrics = OrderedDict()
            for gkey in tvars['gd']:
               metrics[gkey] = []
            for dkey in tvars['dd']:
               metrics[dkey] = []   

            # Loop across all batches in chunk
            for bi in xrange(num_batches):
               
               
                # Train and record metrics
                if itr % (update_ratio+1)==0:
                    gen_out = tfuncs['update_gen'](bi)
                    for key,entry in zip(tvars['gd'],gen_out):
                        metrics[key].append(entry)            
                else:
                    d_out = tfuncs['update_discrim'](bi)
                    for key,entry in zip(tvars['dd'],d_out):
                        metrics[key].append(entry)
                    

                
                
                itr += 1    
                
            for key in metrics:
                metrics[key] = float(np.mean(metrics[key]))
      
            # Chunk-wise metrics
            if (iter_counter-1) % 50 ==0:            
                title = 'epoch   itr    '
                form = []
                for item in metrics:
                    title = title+'  '+str(item)
                    form.append(len(str(item)))
                    
                logging.info(title)
            log_output = '%4d '%epoch + '%6d  '%itr
            for f,item in zip(form,metrics):
                e = '%'+str(f)+'.4f'
                log_output = log_output+'  '+e%metrics[item]   
            logging.info(log_output)
            # logging.info('epoch: {:4d}, itr: {:8d}, ag_loss: {:7.4f}, adg_loss: {:7.4f}, add_loss: {:7.4f}, acc: {:5.3f}, ploss: {:7.4f}, pacc: {:5.3f}'.format(epoch,itr,agloss,adgloss,addloss,accuracy,ploss,pixel_accuracy))
            mlog.log(epoch=epoch,itr=itr,metrics=metrics)
            # Log Chunk Metrics


                      
        # If we see improvement, save weights and produce output images
        # if cfg['reconstruct'] or cfg['introspect']:
        if not (epoch%cfg['checkpoint_every_nth']):                              
        
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
            plot_image_grid(images,6,9,'pics/'+str(args.config_path)[:-3]+'_'+str(epoch)+'.png')
            
            # Close test set
            test_set.close(state=None)
            
            # Save weights
            params = list(set(lasagne.layers.get_all_params(model['l_out'],trainable=True)+\
                              lasagne.layers.get_all_params(model['l_discrim'],trainable=True)+\
                              [x for x in lasagne.layers.get_all_params(model['l_out'])+\
                                lasagne.layers.get_all_params(model['l_discrim'])if x.name[-4:]=='mean' or x.name[-7:]=='inv_std']))
            GANcheckpoints.save_weights(weights_fname, params,{'epoch':epoch,'itr': itr, 'ts': time.time(),'learning_rate':np.float32(tvars['learning_rate'].get_value())})
            
    logging.info('training done')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='config .py file')
    parser.add_argument('--resume',type=bool,default=False)
    args = parser.parse_args()
    main(args)
