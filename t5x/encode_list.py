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

queries = [
    'A blue bicycle parked by a metal gate.',
    'an orange bike leaning on a pole in the snow',
    'There are some colored lights hanging from street lamps',
    'The apple symbol is show on the Premacy car.',
    'An orange motorcycle is shown at close range.',
    'A blue motorbike has a "Minnesota" license plate.',
    'The side of an American aircraft showing the door.',
    'A floor drain is set in concrete with an advisory not to step on it.',
    'a bus covered with assorted colorful graffiti on the side of it',
    'A bus covered in graffiti is stationary on the pavement',
    'A bike image on some double doors with windows.',
    'A red train cart is shown at close range.',
    'A truck is drying several items of clothing in the sun.',
    'The rotted out bed of a truck left in the woods.',
    'A boat on the water tied down to a stake.',
    'An inflatable raft that has its top open.',
    'A red light in front of a tall building',
    "A street sign says Walk and Don't Walk.",
    'A red and blue fire hydrant with flowers around it.',
    'A red fire hydrant with an open sign on it',
    'A street sign with stickers on the back of it.',
    'A red stop sign with lots of writing all over it.',
    'A parking meter with a time expired label on it.',
    'A blue faced machine for printing parking passes. ',
    'A bag full of trash sitting on a old park bench.',
    'A park bench sits under a tree with the sun shining.',
    'a picture of a flamingo scratching its neck',
    'A large blue bird standing next to a painting of flowers.',
    "A cat is staring ahead as the back another cat's head is seen in front of him.",
    'An orange cat looking upside down through glasses.',
    'A bulldog is wearing a black pirate hat.',
    'a dog standing at a gate wanting to get out of the fence ',
    'A brown and white horse wearning a harness eating some hay',
    'A brown and white horse is wearing a blue muzzle.',
    'A sheep standing in the grass with something on its ears. ',
    'A sheep looking through the slats of a wired fence.',
    'A black cow looks directly at the camera.',
    'A horned cow  standing in a green grass field. ',
    'An elephant with trimmed tusks relaxing with a covering of hay on its back.',
    'An elephant placing some leaves in its mouth with its trunk.',
    'The polar bear swimming briskly through the ocean current',
    'A large black bear is facing straight ahead.',
    'The painting is of a zebra in a cage. ',
    'A ZEBRA IS EATING GRASS ON THE GROUND ',
    "A giraffe leaning it's long neck over the fence to eat leaves off a bush.",
    'a very big giraffe that is siting on the ground',
    'A stuffed animal in a bag in a room.',
    'A large umbrella open wide on a pole.',
    'The plants can be seen through the orange mesh.',
    'A fireplace mantle that has been faced in a light stone.',
    'A tan table top hosts a pen and a necktie.',
    'A gold tie is tied under a brown dress shirt with stripes.',
    'A piece of gray luggage with travel stickers.',
    'A cat relaxes in a suitcase next to a pile of clothes.',
    'A yellow frisbee next to a box with Nike Cleats.',
    'Blue Frisbee and envelope it was shipped in. ',
    'A large pair of skis rests against a wall.',
    'APPEARS TO BE SOME OLD SKIS PROPPED UP AGAINST A STONE MEMORIAL',
    'A snowboard standing upright in a snow bank.',
    'A snowboard and gloves laying in the snow. ',
    'A blue and white traffic sign on a grey brick wall.',
    'A cat shaped kite sits in the grass. ',
    'there is a very colorful kite that is in the air',
    'A bat and shin guard in the closet.',
    'a baseball bat with a batting helmet upsidedown',
    'A stuffed animal that is frowning is on a skateboard. ',
    'An very well used upside down skateboard on grass',
    'White surfboard leaning against a brown tiki wall.',
    "A small surfboard sign that says TRADER VIC'S, Los Angeles.",
    'A couple of snowmen have been built in suburban backyards after a recently fallen snow. ',
    'Very large tennis racket with hello kitty on it.',
    'A blender and jar of red liquid on a table.',
    'A plastic jar of honey glowing in the middle of the dark.',
    'a table with a blender and a glass on it ',
    'A glass of wine sitting on the top of a swimming pool side.',
    'a glass measuring cup with yellow liquid in it',
    'a blender full of liquid is spilling everywhere',
    'a dish of food topped with sour cream and a fork',
    'A pizza and fork on a tray on the table',
    'A knife sitting on top of a wooden table next to a knife.',
    "Beet tops and a chef's knife on a cutting board.",
    'A pile of cabbage, noodles, and meat next to chopsticks',
    'Blender half full of a slurry, next to other electrical appliances.',
    'A bowl of a meal with an egg, "sunny side up", laid on top of everything',
    'A pork dish with onions and peppers on a white plate.',
    'Someone wrote a message on a bunch of bananas.',
    'Fruit growing on the side of a tree in a jungle. ',
    'A bunch of apples stacked on a plate',
    'Apples on tree ready to pick in garden area.',
    'A sandwich with meat, vegetables, peppers, and lettuce.',
    'A pile of crab is seasoned and well cooked',
    'Cut up blood red oranges lay on a blue surface.',
    'A plate of oranges sliced on top of a table',
    'A pile of broccoli laying on a plastic cutting board.',
    'A plate of food has noodles and broccoli.',
    'A tray that has meat and carrots on a table.',
    'a stuffed grey rabbit holding a pretend carrot',
    'The two hotdogs have brown mustard on them. ',
    'a plate with a couple of hot dogs on it ',
    'View of what could possibly be a pizza with colorful vegetables as toppings.',
    'A pizza that is covered in a lot of toppings.',
    'A picture of a glazed donut with meat in the middle.',
    'A donut is covered with glaze and sprinkles.',
    'Fresh red strawberries on a whipped dessert. ',
    'Colorful icing on a pastry in the shapes of flowers. ',
    'The cat is sleeping comfortably on the chair. ',
    'A white cat curled up on a wooden chair.',
    'A dog is sleeping on a pile of pillows.',
    'a girl laying on the couch while on an laptop',
    'A bouguet of wilted red roses on a table.',
    'A small green vase displays some small yellow blooms.',
    'A tear in a black, blue, white and yellow piece of material.',
    'A bed with white comforter and two black stiletto heels.',
    'A plate of food consisting of rice, meat and vegetables.',
    'A plate of food is centered around a portion of rice',
    'a porcelian scuplture sitting on a table next to a cup',
    'A bucket collects drips of water, inside a metal basin.',
    'The back of a flat screened tv connected to a ipad .',
    'A cluster of pine trees are in a barren area.',
    'A bobble head is placed on a laptop keyboard',
    'A computer mouse sitting on a keyboard on a desk.',
    'A silver and black wired computer mouse on a wooden surface.',
    'A slug crawling on the ground around flower petals.',
    'two remotes sitting on a table together ',
    'The wii controller remote is turned on and has one light showing.',
    'A black computer keyboard with a bunch of keys on it',
    'A keyboard that is missing some keys in the bottom row',
    'A cellphone standing upright with its camera side facing forward.',
    'There is a cell phone on a table.',
    'A microwave with fake eyes and a beard on it',
    'Food steaming machine on the shelf of a store.',
    'A yellow shallow baking dish in an open oven.',
    'some food cooking on trays in an oven',
    'The toaster oven is turned on, on the counter in the kitchen.',
    'A picture of a toaster plastered on a do not enter sign',
    'a old kitchen with no appliances inside of it ',
    'Two pictures of a hole in a counter with a lid.',
    'a fridge is slightly open in a room',
    'The poster board is a special place for remembrances.',
    'A stuffed animal sitting in the grass, with a book in front of it.',
    'A run down windows with a broken fence outside.',
    'An old and rusted clock is mounted on a brick wall.',
    'a big metal clock sitting on the wall by itself',
    'Bouquet of flowers sitting in a clear glass vase. ',
    'A blue jug in a garden filled with mud.  ',
    'a sign that is advertising a barber shop',
    'A pair of scallop scrapbooking scissors on top of an envelope.',
    'A stuffed bear is wearing a shirt with personalized writing.',
    'A pair of yellow and pink teddy bears leaning against a wall.',
    'A large statue of a female cow with a blonde wig',
    'Rubber shoes lying on a carpet with floral prints',
    'A frog with teeth is on a purple toothbrush.',
    'A worn toothbrush sits on a windowsill near the screen.',
]

# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

AUTOTUNE = tf.data.experimental.AUTOTUNE

def print_model_size(params):
    model_params_size = jax.tree_map(lambda x: x.size, params)
    total_params_size = sum(jax.tree_flatten(model_params_size)[0])
    print('model parameter count:', total_params_size)


def encode_list(
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
  tokens = tokenizer.encode_tf(queries)
  tokens = tf_text.pad_model_inputs(tokens, 512)[0].numpy()

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

  np.savez_compressed('precomputed_queries.npz', text=queries, feature=features)

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
    encode_list_using_gin = gin.configurable(encode_list)

    gin_utils.parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings)

    encode_list_using_gin()


  gin_utils.run(main)
