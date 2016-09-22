# Neural-Photo-Editor
A simple interface for editing natural photos with generative neural networks.

![GUI](http://i.imgur.com/w1U20EI.png)

This repository contains code for the paper ["Neural Photo Editing with Introspective Adversarial Networks,"] (arXiv link coming tonight) and the [Associated Video](https://www.youtube.com/watch?v=FDELBFSeqQs).

## Installation
To run the Neural Photo Editor, you will need:

- [Theano](http://deeplearning.net/software/theano/) 
- [lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html)
- I highly recommend [cuDNN](https://developer.nvidia.com/cudnn) as speed is key. You'll need to uncomment my explicit DNN calls if you wish to not use it.
- numpy, scipy, PIL, Tkinter and tkColorChooser, but it is likely that your python distribution already has those.

## Running the NPE
I've included a pre-trained model (IAN_simple.npz), a slimmed-down version of the IAN without MDC or RGB-Beta blocks, which runs with 0 lag on a laptop GPU with <1GB (GT730M).

```sh
python NPE.py
```
If you wish to use a different model, simply edit the line with "config path" in the NPE.py file.

## Commands:
- You can paint the image by picking a color and painting on the image, or paint in the latent space canvas (the red and blue tiles below the image). 
- The long horizontal slider controls the magnitude of the latent brush, and the smaller horizontal slider controls the size of both the latent and the main image brush.
- You can select different entries from the subset of the celebA validation set (included in this repository as an .npz) by typing in a number from 0-999 in the bottom left box and hitting "infer."
- Use the reset button to return to the last inferred result.
- Use the sample button to generate a random latent vector and corresponding image.
- Use the scroll wheel to lighten or darken an image patch (equivalent to using a pure white or pure black paintbrush). Note that this automatically returns you to sample mode, and may require hitting "infer" rather than "reset" to get back to photo editing.


## Training an IAN on celebA
You will need [Fuel](https://github.com/mila-udem/fuel) along with the 64x64 version of celebA, [instructions for acquiring which are here](https://github.com/vdumoulin/discgen). 

If you want to plot latent space mappings, you will need [matplotlib](http://matplotlib.org/).

If you wish to train a model, the IAN.py file contains the model configuration, and the train_IAN.py file contains the training code, which can be run like this:

```sh
python train_IAN.py IAN.py
```

By default, this code will save (and overwrite!) the weights to a .npz file with the same name as the config.py file (i.e. "IAN.py -> IAN.npz"), and will output a jsonl log of the training with metrics recorded after every chunk (a chunk being a set of minibatches loaded into shared memory).
Use the --resume=True flag when calling to resume training a model--it will automatically pick up with the same LR and epoch it was last saved at.

## Sampling the IAN
#
You can evaluate a classifier's performance on the ModelNet40 dataset, averaging predictions over 12 rotations, with:

```sh
python sample_IAN.py IAN.py
```
## Known Bugs:
Occasionally the color picker freaks out and returns "None"s rather than values. Simply re-choose a color to fix this.


## Notes
More pre-trained models and the remainder of the IAN experiments (including SVHN) coming soon.

## Acknowledgments
This code contains lasagne layers and other goodies adopted from a number of places:
- Layers that are not my own creation should be appropriately attributed here
- MADE wrapped from the implementation by M. Germain et al: https://github.com/mgermain/MADE
- Gaussian Sample layer from Tencia Lee's Recipe: https://github.com/Lasagne/Recipes/blob/master/examples/variational_autoencoder/variational_autoencoder.py
- Minibatch Discrimination layer from OpenAI's Improved GAN Techniques: https://github.com/openai/improved-gan
- Deconv Layer adapted from Radford's DCGAN: https://github.com/Newmu/dcgan_code
- Image-Grid Plotter adopted from Lamb's Discriminative Regularization: https://github.com/vdumoulin/discgen

