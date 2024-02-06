# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Library imports for ClusterResolvers.

  This library contains all implementations of ClusterResolvers.
  ClusterResolvers are a way of specifying cluster information for distributed
  execution. Built on top of existing `ClusterSpec` framework, ClusterResolvers
  are a way for TensorFlow to communicate with various cluster management
  systems (e.g. GCE, AWS, etc...).

"""

import sys as _sys

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import UnionClusterResolver as UnionResolver
from tensorflow.python.distribute.cluster_resolver.gce_cluster_resolver import GCEClusterResolver
from tensorflow.python.distribute.cluster_resolver.kubernetes_cluster_resolver import KubernetesClusterResolver
from tensorflow.python.distribute.cluster_resolver.slurm_cluster_resolver import SlurmClusterResolver
from tensorflow.python.distribute.cluster_resolver.tfconfig_cluster_resolver import TFConfigClusterResolver
from tensorflow.python.distribute.cluster_resolver.tpu.tpu_cluster_resolver import TPUClusterResolver