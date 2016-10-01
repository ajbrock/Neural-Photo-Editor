# Plat Interface for Convenience and Interoperability
# Adopted from Plat by Tom White : https://github.com/dribnet/plat

import theano
import theano.tensor as T
import lasagne
import imp
import GANcheckpoints

# Generic class for using IAN style models with the NPE.
class IAN:
    def __init__(self, config_path,dnn):
        """
        Initializate class give either a filename or a model
        Usually this method will load a model from disk and store internally,
        but model can also be provided directly instead (useful when training)
        """
        config_module = imp.load_source('config',config_path)
        self.cfg = config_module.cfg
        self.weights_fname = str(config_path)[:-3]+'.npz'
        self.model = config_module.get_model(dnn=dnn)
        
        # Load weights
        print('Loading weights')                                                                                     
        params = list(set(lasagne.layers.get_all_params(self.model['l_out'],trainable=True)+\
                 lasagne.layers.get_all_params(self.model['l_discrim'],trainable=True)+\
                 [x for x in lasagne.layers.get_all_params(self.model['l_out'])+\
                 lasagne.layers.get_all_params(self.model['l_discrim'])\
                 if x.name[-4:]=='mean' or x.name[-7:]=='inv_std']))
        GANcheckpoints.load_weights(self.weights_fname,params)
    
        # Shuffle weights if using IAF with MADE
        if 'l_IAF_mu' in self.model:
            print ('Shuffling MADE masks')
            self.model['l_IAF_mu'].reset("Once")
            self.model['l_IAF_ls'].reset("Once")
        
        print('Compiling Theano Functions')
        # Input Tensor
        self.X = T.TensorType('float32', [False]*4)('X')
        
        # Latent Vector
        self.Z = T.TensorType('float32', [False]*2)('Z')
        
        # X_hat(Z)
        self.X_hat = lasagne.layers.get_output(self.model['l_out'],{self.model['l_Z']:self.Z},deterministic=True)
        self.X_hat_fn = theano.function([self.Z],self.X_hat)
        
        # Z_hat(X)
        self.Z_hat=lasagne.layers.get_output(self.model['l_Z'],{self.model['l_in']:self.X},deterministic=True)
        self.Z_hat_fn = theano.function([self.X],self.Z_hat) 
        
        # Imgrad Functions
        r1,r2 = T.scalar('r1',dtype='int32'),T.scalar('r2',dtype='int32')
        c1,c2 = T.scalar('c',dtype='int32'),T.scalar('c2',dtype='int32')
        RGB = T.tensor4('RGB',dtype='float32')
        
        # Image Gradient Function, evaluates the change in latents which would lighten the image in the local area
        self.calculate_lighten_gradient = theano.function([c1,r1,c2,r2,self.Z],T.grad(T.mean(self.X_hat[0,:,r1:r2,c1:c2]),self.Z))
        
        # Image Color Gradient Function, evaluates the change in latents which would push the image towards the local desired RGB value
        # Consider changing this to only take in a smaller RGB array, rather than a full-sized, indexed RGB array.
        # Also consider using the L1 loss instead of L2
        self.calculate_RGB_gradient = theano.function([c1,r1,c2,r2,RGB,self.Z],T.grad(T.mean((T.sqr(-self.X_hat[0,:,r1:r2,c1:c2]+RGB[0,:,r1:r2,c1:c2]))),self.Z)) # may need a T.mean
        
    def imgrad(self,c1,r1,c2,r2,z):
        """
        Calculate the change in latents which would lighten the local image patch.
        """
        return self.calculate_lighten_gradient(c1,r1,c2,r2,z)
        
    def imgradRGB(self,c1,r1,c2,r2,RGB,z):
        """
        Calculate the change in latents which would move the local image patch towards the RGB value of RGB.
        """
        return self.calculate_RGB_gradient(c1,r1,c2,r2,RGB,z)    
        
    def encode_images(self, images):
        """
        Encode images x => z
        images is an n x 3 x s x s numpy array where:
          n = number of images
          3 = R G B channels
          s = size of image (eg: 64, 128, etc)
          pixels values for each channel are encoded [-1,1]
        returns an n x z numpy array where:
          n = len(images)
          z = dimension of latent space
        """        
        return self.Z_hat_fn(images)

    def get_zdim(self):
        """
        Returns the integer dimension of the latent z space
        """
        return self.cfg['num_latents']

    def sample_at(self, z):
        """
        Decode images z => x
        z is an n x z numpy array where:
          n = len(images)
          z = dimension of latent space
        return images as an n x 3 x s x s numpy array where:
          n = number of images
          3 = R G B channels
          s = size of image (eg: 64, 128, etc)
          pixels values for each channel are encoded [-1,1]
        """
        return self.X_hat_fn(z)