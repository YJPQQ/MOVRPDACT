# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Conversion of eager-style Python into TensorFlow graph code.

NOTE: In TensorFlow 2.0, AutoGraph is automatically applied when using
`tf.function`. This module contains lower-level APIs for advanced use.

AutoGraph transforms a subset of Python which operates on TensorFlow objects
into equivalent TensorFlow graph code. When executing the graph, it has the same
effect as if you ran the original code in eager mode.
Python code which doesn't operate on TensorFlow objects remains functionally
unchanged, but keep in mind that `tf.function` only executes such code at trace
time, and generally will not be consistent with eager execution.

For more information, see the
[AutoGraph reference documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/index.md),
and the [tf.function guide](https://www.tensorflow.org/guide/function#autograph_transformations).

"""

import sys as _sys

from . import experimental
from tensorflow.python.autograph.impl.api import to_code
from tensorflow.python.autograph.impl.api import to_graph
from tensorflow.python.autograph.utils.ag_logging import set_verbosity
from tensorflow.python.autograph.utils.ag_logging import trace