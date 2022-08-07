# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=line-too-long
# pyformat: disable
r"""This script runs inference on a T5X-compatible model.

"""
# pyformat: enable
# pylint:enable=line-too-long

import time
import concurrent.futures
import functools
import hashlib
import json
import os
import os.path as osp
import re
import shutil
import time
from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence, Tuple, Type

# TODO(adarob): Re-enable once users are notified and tests are updated.
# Must be set before flax imports.
# pylint:disable=g-import-not-at-top
os.environ['FLAX_LAZY_RNG'] = 'no'
from absl import logging
from clu import metric_writers
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import seqio
import argparse
from t5x import gin_utils
from t5x import models
from t5x import partitioning
from t5x import utils
from t5x import data
import tensorflow as tf
from tensorflow.io import gfile
from typing_extensions import Protocol
import io
import multiprocessing as mp
from tqdm import tqdm
import seqio
import tensorflow_text as tf_text


# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

AUTOTUNE = tf.data.experimental.AUTOTUNE

def print_model_size(params):
    model_params_size = jax.tree_map(lambda x: x.size, params)
    total_params_size = sum(jax.tree_flatten(model_params_size)[0])
    print('model parameter count:', total_params_size)


def encode_json(
    *,
    model: models.BaseTransformerModel,
    restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
    partitioner: partitioning.BasePartitioner,
):
  """Infer function.

  Args:
    mode: Either 'predict' to decode targets, 'score' to compute the log
      likelihood of given targets, or 'predict_with_aux' for both.
    model: The model object to use for inference.
    restore_checkpoint_cfg: Specification for the model parameter checkpoint to
      load.
    partitioner: Partitioner for model parameters and data across devices.
  """
  logging.info('Process ID: %d', jax.process_index())

  fname = 'ssv2_text.json'
  base_name = osp.basename(fname).split('.')[0]
  data = json.load(open(fname, 'r'))

  input_shapes = {'encoder_input_tokens': [1, 512], 'decoder_input_tokens': [1, 62]}
  input_types = {'encoder_input_tokens': jnp.int32, 'decoder_input_tokens': jnp.int32}

  # Initialize optimizer from the existing checkpoint.
  # TODO(adarob): Support inference over multiple checkpoints.
  train_state_initializer = utils.TrainStateInitializer(
      optimizer_def=None,  # Do not load optimizer state.
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      input_types=input_types,
      partitioner=partitioner)
  train_state_axes = train_state_initializer.train_state_axes

  # Disable strictness since we are dropping the optimizer state.
  restore_checkpoint_cfg.strict = False

  train_state = train_state_initializer.from_checkpoint(
      [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0))
  print_model_size(train_state.params)

  def encode_fn(params, encoder_input_tokens):
    return model.module.apply( 
        {'params': params},
        encoder_input_tokens=encoder_input_tokens,
        enable_dropout=False,
        method=model.module.encode
    )

  encode_fn = partitioner.partition(
      encode_fn,
      in_axis_resources=(train_state_axes.params,
                         partitioner.data_partition_spec),
      out_axis_resources=None
  )

  tokenizer = seqio.SentencePieceVocabulary('gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model')
  all_features, all_masks = {}, {}
  for split, texts in data.items():
    print('Computing split:', split)
    tokens = tokenizer.encode_tf(texts)
    tokens, text_mask = tf_text.pad_model_inputs(tokens, 512)
    tokens, text_mask = tokens.numpy(), text_mask.numpy()

    batch_size = 64
    pbar = tqdm(total=len(tokens))
    features = []
    for i in range(0, len(tokens), batch_size):
      inp = tokens[i:i+batch_size]
      n = inp.shape[0]
      inp = np.pad(inp, ((0, batch_size - n), (0, 0)))
      feats_padded = jax.device_get(encode_fn(train_state.params, inp))
      feats_padded = feats_padded[:n]
      features.append(feats_padded)
      pbar.update(n)
    features = np.concatenate(features, axis=0)

    all_features[split] = features
    all_masks[split] = text_mask

  np.savez_compressed(f'{base_name}_features.npz', feature=all_features, mask=text_mask)

  logging.info('DONE')


if __name__ == '__main__':
  # pylint:disable=g-import-not-at-top
  from absl import app
  from absl import flags
  import gin
  # pylint:enable=g-import-not-at-top

  FLAGS = flags.FLAGS

  jax.config.parse_flags_with_absl()

  flags.DEFINE_multi_string(
      'gin_file',
      default=None,
      help='Path to gin configuration file. Multiple paths may be passed and '
      'will be imported in the given order, with later configurations  '
      'overriding earlier ones.')

  flags.DEFINE_multi_string(
      'gin_bindings', default=[], help='Individual gin bindings.')

  flags.DEFINE_list(
      'gin_search_paths',
      default=['.'],
      help='Comma-separated list of gin config path prefixes to be prepended '
      'to suffixes given via `--gin_file`. If a file appears in. Only the '
      'first prefix that produces a valid path for each suffix will be '
      'used.')

  def main(argv: Sequence[str]):
    """Wrapper for pdb post mortems."""
    _main(argv)

  def _main(argv: Sequence[str]):
    """True main function."""
    if len(argv) > 1:
      raise app.UsageError('Too many command-line arguments.')

    # Create gin-configurable version of `infer`.
    encode_json_using_gin = gin.configurable(encode_json)

    gin_utils.parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings)

    encode_json_using_gin()


  gin_utils.run(main)
