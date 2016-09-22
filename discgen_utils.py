# Plot Image Grid function imported from Discriminative Regularization for Generative Models by Lamb et al:
# https://github.com/vdumoulin/discgen
import six
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm, pyplot
from mpl_toolkits.axes_grid1 import ImageGrid



def plot_image_grid(images, num_rows, num_cols, save_path=None):
    """Plots images in a grid.

    Parameters
    ----------
    images : numpy.ndarray
        Images to display, with shape
        ``(num_rows * num_cols, num_channels, height, width)``.
    num_rows : int
        Number of rows for the image grid.
    num_cols : int
        Number of columns for the image grid.
    save_path : str, optional
        Where to save the image grid. Defaults to ``None``,
        which causes the grid to be displayed on screen.

    """
    figure = pyplot.figure()
    grid = ImageGrid(figure, 111, (num_rows, num_cols), axes_pad=0.1)

    for image, axis in zip(images, grid):
        axis.imshow(image.transpose(1, 2, 0), interpolation='nearest')
        axis.set_yticklabels(['' for _ in range(image.shape[1])])
        axis.set_xticklabels(['' for _ in range(image.shape[2])])
        axis.axis('off')

    if save_path is None:
        pyplot.show()
    else:
        pyplot.savefig(save_path, transparent=True, bbox_inches='tight',dpi=212)
        pyplot.close()