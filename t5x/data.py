import glob
import os.path as osp
import io
import tarfile
import numpy as np
from PIL import Image
import jax
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import tensorflow_text as tf_text
import seqio
from flax import jax_utils


GCS_PATH = 'gs://imagen/datasets'


def load_laion(config, train):
    split = 'train' if train else 'test'
    folder = osp.join(config.data_path,  split, '*.tar')
    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
    else:
        fns = list(glob.glob(folder))
    fns.sort()
    
    tokenizer = seqio.SentencePieceVocabulary('gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model')

    def read(path):
        path = io.BytesIO(file_io.FileIO(path, 'rb').read())
        tar = tarfile.open(fileobj=path)

        images, texts = [], []
        for file in tar.getmembers():
            name = file.name
            content = tar.extractfile(file).read()
            if name.endswith('.txt'):
                text = content
                texts.append(text)
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

    def tokenize(texts):
        texts = tf.ensure_shape(texts, (None,))
        texts = tokenizer.encode_tf(texts)

        texts = texts[:, :config.max_sequence_length - 1]
        eos = tf.fill([tf.shape(texts)[0], 1], tokenizer.eos_id)
        texts = tf.concat([texts, eos], axis=1)
        texts, _ = tf_text.pad_model_inputs(texts, config.max_sequence_length, pad_value=tokenizer.pad_id)
        return texts

    dataset = dataset.map(tokenize)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, fns
 
