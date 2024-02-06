# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Sparse Tensor Representation.

See also `tf.sparse.SparseTensor`.

"""

import sys as _sys

from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.ops.array_ops import sparse_mask as mask
from tensorflow.python.ops.array_ops import sparse_placeholder as placeholder
from tensorflow.python.ops.bincount_ops import sparse_bincount as bincount
from tensorflow.python.ops.data_flow_ops import SparseConditionalAccumulator
from tensorflow.python.ops.math_ops import sparse_segment_mean as segment_mean
from tensorflow.python.ops.math_ops import sparse_segment_sqrt_n as segment_sqrt_n
from tensorflow.python.ops.math_ops import sparse_segment_sum as segment_sum
from tensorflow.python.ops.sparse_ops import from_dense
from tensorflow.python.ops.sparse_ops import sparse_add as add
from tensorflow.python.ops.sparse_ops import sparse_concat as concat
from tensorflow.python.ops.sparse_ops import sparse_cross as cross
from tensorflow.python.ops.sparse_ops import sparse_cross_hashed as cross_hashed
from tensorflow.python.ops.sparse_ops import sparse_expand_dims as expand_dims
from tensorflow.python.ops.sparse_ops import sparse_eye as eye
from tensorflow.python.ops.sparse_ops import sparse_fill_empty_rows as fill_empty_rows
from tensorflow.python.ops.sparse_ops import sparse_maximum as maximum
from tensorflow.python.ops.sparse_ops import sparse_merge as merge
from tensorflow.python.ops.sparse_ops import sparse_minimum as minimum
from tensorflow.python.ops.sparse_ops import sparse_reduce_max as reduce_max
from tensorflow.python.ops.sparse_ops import sparse_reduce_max_sparse as reduce_max_sparse
from tensorflow.python.ops.sparse_ops import sparse_reduce_sum as reduce_sum
from tensorflow.python.ops.sparse_ops import sparse_reduce_sum_sparse as reduce_sum_sparse
from tensorflow.python.ops.sparse_ops import sparse_reorder as reorder
from tensorflow.python.ops.sparse_ops import sparse_reset_shape as reset_shape
from tensorflow.python.ops.sparse_ops import sparse_reshape as reshape
from tensorflow.python.ops.sparse_ops import sparse_retain as retain
from tensorflow.python.ops.sparse_ops import sparse_slice as slice
from tensorflow.python.ops.sparse_ops import sparse_softmax as softmax
from tensorflow.python.ops.sparse_ops import sparse_split as split
from tensorflow.python.ops.sparse_ops import sparse_tensor_dense_matmul as matmul
from tensorflow.python.ops.sparse_ops import sparse_tensor_dense_matmul as sparse_dense_matmul
from tensorflow.python.ops.sparse_ops import sparse_tensor_to_dense as to_dense
from tensorflow.python.ops.sparse_ops import sparse_to_indicator as to_indicator
from tensorflow.python.ops.sparse_ops import sparse_transpose as transpose