# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Utilities for text input preprocessing.

Deprecated: `tf.keras.preprocessing.text` APIs are not recommended for new code.
Prefer `tf.keras.utils.text_dataset_from_directory` and
`tf.keras.layers.TextVectorization` which provide a more efficient approach
for preprocessing text input. For an introduction to these APIs, see
the [text loading tutorial]
(https://www.tensorflow.org/tutorials/load_data/text)
and [preprocessing layer guide]
(https://www.tensorflow.org/guide/keras/preprocessing_layers).

"""

import sys as _sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import tokenizer_from_json