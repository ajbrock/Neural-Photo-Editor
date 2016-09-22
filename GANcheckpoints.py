
import logging
import cPickle as pickle
import warnings
import numpy as np

from path import Path

import lasagne

def save_weights(fname, params, metadata=None):
    """ assumes all params have unique names.
    """
    # Includes batchnorm params now
    names = [par.name for par in params]
    if len(names) != len(set(names)):
        raise ValueError('need unique param names')
    param_dict = { param.name : param.get_value(borrow=False)
            for param in params }
    if metadata is not None:
        param_dict['metadata'] = pickle.dumps(metadata)
    logging.info('saving {} parameters to {}'.format(len(params), fname))
    # try to avoid half-written files
    fname = Path(fname)
    if fname.exists():
        tmp_fname = Path(fname.stripext() + '.tmp.npz') # TODO yes, this is a hack
        np.savez_compressed(str(tmp_fname), **param_dict)
        tmp_fname.rename(fname)
    else:
        np.savez_compressed(str(fname), **param_dict)


def load_weights(fname, params):
    # params = lasagne.layers.get_all_params(l_out,trainable=True)+[log_sigma]+[x for x in lasagne.layers.get_all_params(l_out) if x.name[-4:]=='mean' or x.name[-7:]=='inv_std']
    names = [ par.name for par in params ]
    if len(names)!=len(set(names)):
        raise ValueError('need unique param names')

    param_dict = np.load(fname)
    for param in params:
        if param.name in param_dict:
            stored_shape = np.asarray(param_dict[param.name].shape)
            param_shape = np.asarray(param.get_value().shape)
            if not np.all(stored_shape == param_shape):
                warn_msg = 'shape mismatch:'
                warn_msg += '{} stored:{} new:{}'.format(param.name, stored_shape, param_shape)
                warn_msg += ', skipping'
                warnings.warn(warn_msg)
            else:
                param.set_value(param_dict[param.name])
        else:
            logging.warn('unable to load parameter {} from {}'.format(param.name, fname))
    if 'metadata' in param_dict:
        metadata = pickle.loads(str(param_dict['metadata']))
    else:
        metadata = {}
    return metadata
