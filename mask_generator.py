## Mask generator from MADE: https://github.com/mgermain/MADE

import copy
import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams  # Limited but works on GPU
from theano.tensor.shared_randomstreams import RandomStreams
# from theano.gpuarray.dnn import GpuDnnSoftmax as mysoftmax

def mysoftmax(x):
    e_x = T.exp(x - x.max())
    return e_x / e_x.sum()

class MaskGenerator(object):

    def __init__(self, input_size, hidden_sizes, l, random_seed=1234):
        self._random_seed = random_seed
        self._mrng = MRG_RandomStreams(seed=random_seed)
        self._rng = RandomStreams(seed=random_seed)

        self._hidden_sizes = hidden_sizes
        self._input_size = input_size
        self._l = l

        self.ordering = theano.shared(value=np.arange(input_size, dtype=theano.config.floatX), name='ordering', borrow=False)

        # Initial layer connectivity
        self.layers_connectivity = [theano.shared(value=(self.ordering + 1).eval(), name='layer_connectivity_input', borrow=False)]
        for i in range(len(self._hidden_sizes)):
            self.layers_connectivity += [theano.shared(value=np.zeros((self._hidden_sizes[i]), dtype=theano.config.floatX), name='layer_connectivity_hidden{0}'.format(i), borrow=False)]
        self.layers_connectivity += [self.ordering]

        ## Theano functions
        new_ordering = self._rng.shuffle_row_elements(self.ordering)
        self.shuffle_ordering = theano.function(name='shuffle_ordering',
                                                inputs=[],
                                                updates=[(self.ordering, new_ordering), (self.layers_connectivity[0], new_ordering + 1)])

        self.layers_connectivity_updates = []
        for i in range(len(self._hidden_sizes)):
            self.layers_connectivity_updates += [self._get_hidden_layer_connectivity(i)]
        # self.layers_connectivity_updates = [self._get_hidden_layer_connectivity(i) for i in range(len(self._hidden_sizes))]  # WTF THIS DO NOT WORK
        self.sample_connectivity = theano.function(name='sample_connectivity',
                                                   inputs=[],
                                                   updates=[(self.layers_connectivity[i+1], self.layers_connectivity_updates[i]) for i in range(len(self._hidden_sizes))])

        # Save random initial state
        self._initial_mrng_rstate = copy.deepcopy(self._mrng.rstate)
        self._initial_mrng_state_updates = [state_update[0].get_value() for state_update in self._mrng.state_updates]

        # Ensuring valid initial connectivity
        self.sample_connectivity()

    def reset(self):
        # Set Original ordering
        self.ordering.set_value(np.arange(self._input_size, dtype=theano.config.floatX))

        # Reset RandomStreams
        self._rng.seed(self._random_seed)

        # Initial layer connectivity
        self.layers_connectivity[0].set_value((self.ordering + 1).eval())
        for i in range(1, len(self.layers_connectivity)-1):
            self.layers_connectivity[i].set_value(np.zeros((self._hidden_sizes[i-1]), dtype=theano.config.floatX))
        self.layers_connectivity[-1].set_value(self.ordering.get_value())

        # Reset MRG_RandomStreams (GPU)
        self._mrng.rstate = self._initial_mrng_rstate
        for state, value in zip(self._mrng.state_updates, self._initial_mrng_state_updates):
            state[0].set_value(value)

        self.sample_connectivity()

    def _get_p(self, start_choice):
        start_choice_idx = (start_choice-1).astype('int32')
        p_vals = T.concatenate([T.zeros((start_choice_idx,)), (self._l * T.arange(start_choice, self._input_size, dtype=theano.config.floatX))])
        p_vals = T.inc_subtensor(p_vals[start_choice_idx], 1.)  # Stupid hack because de multinomial does not contain a safety for numerical imprecision.
        return p_vals

    def _get_hidden_layer_connectivity(self, layerIdx):
        layer_size = self._hidden_sizes[layerIdx]
        if layerIdx == 0:
            p_vals = self._get_p(T.min(self.layers_connectivity[layerIdx]))
        else:
            p_vals = self._get_p(T.min(self.layers_connectivity_updates[layerIdx-1]))

        # #Implementations of np.choose in theano GPU
        # return T.nonzero(self._mrng.multinomial(pvals=[self._p_vals] * layer_size, dtype=theano.config.floatX))[1].astype(dtype=theano.config.floatX)
        # return T.argmax(self._mrng.multinomial(pvals=[self._p_vals] * layer_size, dtype=theano.config.floatX), axis=1)
        return T.sum(T.cumsum(self._mrng.multinomial(pvals=T.tile(p_vals[::-1][None, :], (layer_size, 1)), dtype=theano.config.floatX), axis=1), axis=1)

    def _get_mask(self, layerIdxIn, layerIdxOut):
        return (self.layers_connectivity[layerIdxIn][:, None] <= self.layers_connectivity[layerIdxOut][None, :]).astype(theano.config.floatX)

    def get_mask_layer_UPDATE(self, layerIdx):
        return self._get_mask(layerIdx, layerIdx + 1)

    def get_direct_input_mask_layer_UPDATE(self, layerIdx):
        return self._get_mask(0, layerIdx)

    def get_direct_output_mask_layer_UPDATE(self, layerIdx):
        return self._get_mask(layerIdx, -1)