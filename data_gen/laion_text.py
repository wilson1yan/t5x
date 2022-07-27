import os.path as osp
import pandas as pd
import glob

import tensorflow as tf
from tensorflow.python.framework import dtypes


def load_text():
    root = '/home/wilson/data/laion400m/laion400m-data'
    fns = list(glob.glob(osp.join(root, '*.parquet')))

    def read(path):
        data = pd.read_parquet(path)
        texts = data['captions'].tolist()
        return tf.data.Dataset.from_tensor_slices(texts)

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.py_function(
            read, 
            [item],
            [dtypes.string]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.batch(1)
    return dataset

    
dataset = load_text()
print(next(dataset))