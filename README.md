# Neural Photo Editor
A simple interface for editing natural photos with generative neural networks.

![GUI](http://i.imgur.com/w1U20EI.png)

This repository contains code for the paper "[Neural Photo Editing with Introspective Adversarial Networks](http://arxiv.org/abs/1609.07093)," and the [Associated Video](https://www.youtube.com/watch?v=FDELBFSeqQs).

## Installation
To run the Neural Photo Editor, you will need:

- [Theano](http://deeplearning.net/software/theano/), development version.  
- [lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html), development version.
- I highly recommend [cuDNN](https://developer.nvidia.com/cudnn) as speed is key. You'll need to uncomment my explicit DNN calls if you wish to not use it.
NOTE: I am currently working on a non-cuDNN implementation that should allow for non-NVIDIA GPUs (and maybe CPUs too)--stay tuned!
- numpy, scipy, PIL, Tkinter and tkColorChooser, but it is likely that your python distribution already has those.

## Running the NPE
By default, the NPE runs on IAN_simple. This is a slimmed-down version of the IAN without MDC or RGB-Beta blocks, which runs without lag on a laptop GPU with ~1GB of memory (GT730M)

If you're on a Windows machine, you will want to create a .theanorc file and at least set the flag FLOATX=float32. 

If you're on a linux machine, you can just insert THEANO_FLAGS=floatX=float32 before the command line call.

Then, run the command:

```sh
python NPE.py
```
If you wish to use a different model, simply edit the line with "config path" in the NPE.py file. You can make use of any model with an inference mechanism (VAE or ALI-based GAN).

## Commands
- You can paint the image by picking a color and painting on the image, or paint in the latent space canvas (the red and blue tiles below the image). 
- The long horizontal slider controls the magnitude of the latent brush, and the smaller horizontal slider controls the size of both the latent and the main image brush.
- You can select different entries from the subset of the celebA validation set (included in this repository as an .npz) by typing in a number from 0-999 in the bottom left box and hitting "infer."
- Use the reset button to return to the last inferred result.
- Use the sample button to generate a random latent vector and corresponding image.
- Use the scroll wheel to lighten or darken an image patch (equivalent to using a pure white or pure black paintbrush). Note that this automatically returns you to sample mode, and may require hitting "infer" rather than "reset" to get back to photo editing.


## Training an IAN on celebA
You will need [Fuel](https://github.com/mila-udem/fuel) along with the 64x64 version of celebA. See [here](https://github.com/vdumoulin/discgen) for instructions on downloading and preparing it. 

If you wish to train a model, the IAN.py file contains the model configuration, and the train_IAN.py file contains the training code, which can be run like this:

```sh
python train_IAN.py IAN.py
```

By default, this code will save (and overwrite!) the weights to a .npz file with the same name as the config.py file (i.e. "IAN.py -> IAN.npz"), and will output a jsonl log of the training with metrics recorded after every chunk.

Use the --resume=True flag when calling to resume training a model--it will automatically pick up from the most recent epoch.

## Sampling the IAN
#
You can generate a sample and reconstruction+interpolation grid with:

```sh
python sample_IAN.py IAN.py
```

Note that you will need [matplotlib](http://matplotlib.org/). to do so.
## Known Bugs
Occasionally the color picker freaks out and returns "None"s rather than values. Simply re-choose a color to fix this.
My MADE layer currently only accepts hidden unit sizes that are equal to the size of the latent vector, which will present itself as a BAD_PARAM error.
Since the MADE really only acts as an autoregressive randomizer I'm not too worried about this, but it does bare looking into.


## Notes
More pre-trained models (which will be in the repo rather than a drive folder) and the remainder of the IAN experiments (including SVHN and the full pre-trained IAN used to generate the samples from the paper) coming soon.
I'm considering putting up a TensorFlow implementation, if people would be interested in that, let me know (maybe with an Issues 

## Acknowledgments
This code contains lasagne layers and other goodies adopted from a number of places:
- MADE wrapped from the implementation by M. Germain et al: https://github.com/mgermain/MADE
- Gaussian Sample layer from Tencia Lee's Recipe: https://github.com/Lasagne/Recipes/blob/master/examples/variational_autoencoder/variational_autoencoder.py
- Minibatch Discrimination layer from OpenAI's Improved GAN Techniques: https://github.com/openai/improved-gan
- Deconv Layer adapted from Radford's DCGAN: https://github.com/Newmu/dcgan_code
- Image-Grid Plotter adopted from AlexMLamb's Discriminative Regularization: https://github.com/vdumoulin/discgen

