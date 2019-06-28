import numpy as np
import tensorflow as tf
from ray.rllib.models.misc import get_activation_fn, flatten

from ray.rllib.models.visionnet import VisionNetwork, _get_filter_config
from ray.rllib.utils.annotations import override


def tf_normalize(obs, obs_space, low=None, high=None):
    """Result will be float32 tensor with values in [-1, 1]."""
    if low is None:
        low = obs_space.low.flat[0]
    if high is None:
        high = obs_space.high.flat[0]

    mean = (low + high) * 0.5
    if obs_space.dtype != np.float32:
        obs = tf.to_float(obs)

    scaling = 1.0 / (high - mean)
    obs = (obs - mean) * scaling
    return obs


# noinspection PyAbstractClass
class VizdoomVisionNetwork(VisionNetwork):
    @override(VisionNetwork)
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # unpacking Doom observation dict, and
        obs = input_dict['obs']['obs']
        obs = tf_normalize(obs, self.obs_space, low=0, high=255)

        # health, ammo, etc.
        measurements = input_dict['obs']['measurements']

        filters = options.get('conv_filters')
        if not filters:
            filters = _get_filter_config(obs.shape.as_list()[1:])

        activation = get_activation_fn(options.get('conv_activation'))
        fcnet_activation = get_activation_fn(options.get('fcnet_activation'))

        with tf.name_scope('vision_net'):
            for i, (out_size, kernel, stride) in enumerate(filters, 1):
                obs = tf.layers.conv2d(
                    obs,
                    out_size,
                    kernel,
                    stride,
                    activation=activation,
                    padding='same',
                    name='conv{}'.format(i),
                )

            vis_input_flat = flatten(obs)
            all_input = tf.concat([vis_input_flat, measurements], axis=1)

            fc_hiddens = [256]
            for i, fc_hidden in enumerate(fc_hiddens, 1):
                hidden = tf.layers.dense(all_input, fc_hidden, activation=fcnet_activation, name=f'fc{i}')

            # this will be used later for value function
            last_hidden = hidden

            fc_final = tf.layers.dense(last_hidden, num_outputs, activation=None, name=f'fc_final')
            return fc_final, last_hidden