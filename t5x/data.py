import glob
import os.path as osp
import io
import tarfile
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import tensorflow_text as tf_text
import seqio


def load_laion(config):
    folder = osp.join(config.data_path, '*.tar')
    folder_npz = osp.join(config.data_path, '*.npz')
    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
        fns_npz = tf.io.gfile.glob(folder_npz)
    else:
        fns = list(glob.glob(folder))
        fns_npz = list(glob.glob(folder_npz))
    fns.sort()

    with_npz = set([fn[:-4] for fn in fns_npz])
    print('Already computed for {len(with_npz)} files')
    original_len = len(fns)
    fns = [fn for fn in fns if fn[:-4] in with_npz]
    print(f'Computing for {len(fns)} / {original_len}')
    
    tokenizer = seqio.SentencePieceVocabulary('gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model')

    def read(path):
        path = io.BytesIO(file_io.FileIO(path, 'rb').read())
        tar = tarfile.open(fileobj=path)

        files = tar.getmembers()
        files.sort(key=lambda x: x.name)

        texts = []
        for file in files:
            name = file.name
            content = tar.extractfile(file).read()
            if name.endswith('.txt'):
                text = content
                texts.append(text)
        texts = tf.convert_to_tensor(texts, dtype=tf.string)
        return texts, path
        

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read,
            [item],
            [tf.string, tf.string]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    def tokenize(texts, path):
        texts = tf.ensure_shape(texts, (None,))
        texts = tokenizer.encode_tf(texts)

        texts = texts[:, :config.max_sequence_length - 1]
        eos = tf.fill([tf.shape(texts)[0], 1], tokenizer.eos_id)
        texts = tf.concat([texts, eos], axis=1)
        texts, _ = tf_text.pad_model_inputs(texts, config.max_sequence_length, pad_value=tokenizer.pad_id)
        return texts, path

    dataset = dataset.map(tokenize)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
 
