import functools
import os.path as osp
import numpy as np
import pandas as pd
import glob
import seqio

import tensorflow as tf
from tensorflow.python.framework import dtypes


def load_text(split, shuffle_files, seed=None):
    root = '/home/wilson/data/laion400m/laion400m-data'
    fns = list(glob.glob(osp.join(root, '*.parquet')))

    def read(path):
        data = pd.read_parquet(path.decode('utf-8'))
        texts = data['caption'].tolist()
        texts = [t for t in texts if t is not None]
        texts = tf.convert_to_tensor(texts, dtype=tf.string)
        return texts

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read, 
            [item],
            [tf.string]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    def process(texts):
        texts = tf.ensure_shape(texts, (None,))
        return tf.data.Dataset.from_tensor_slices({'tokens': texts})

    dataset = dataset.flat_map(process)
    return dataset


vocabulary = seqio.SentencePieceVocabulary(
    'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model'
)

output_features = {
    'tokens': seqio.Feature(vocabulary=vocabulary)
}


seqio.TaskRegistry.add(
    'laion400m',
    source=seqio.FunctionDataSource(load_text, ['train']),
    preprocessors=[
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos_after_trim
    ],
    output_features=output_features
)

dataset = seqio.get_mixture_or_task('laion400m').get_dataset(
        split='train', shuffle=False, sequence_length={'tokens': None})

total, total_tokens = 0, 0
for ex in dataset.as_numpy_iterator():
    n_tokens = len(ex['tokens'])
    total_tokens += n_tokens
    total += 1

    print(total_tokens / total)
