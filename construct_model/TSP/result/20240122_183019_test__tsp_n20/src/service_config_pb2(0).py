# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/service_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow.core.protobuf import data_service_pb2 as tensorflow_dot_core_dot_protobuf_dot_data__service__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/core/protobuf/service_config.proto',
  package='tensorflow.data.experimental',
  syntax='proto3',
  serialized_options=_b('ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto'),
  serialized_pb=_b('\n-tensorflow/core/protobuf/service_config.proto\x12\x1ctensorflow.data.experimental\x1a+tensorflow/core/protobuf/data_service.proto\"\x8d\x02\n\x10\x44ispatcherConfig\x12\x0c\n\x04port\x18\x01 \x01(\x03\x12\x10\n\x08protocol\x18\x02 \x01(\t\x12\x10\n\x08work_dir\x18\x03 \x01(\t\x12\x1b\n\x13\x66\x61ult_tolerant_mode\x18\x04 \x01(\x08\x12\x18\n\x10worker_addresses\x18\x07 \x03(\t\x12\x38\n\x0f\x64\x65ployment_mode\x18\t \x01(\x0e\x32\x1f.tensorflow.data.DeploymentMode\x12 \n\x18job_gc_check_interval_ms\x18\x05 \x01(\x03\x12\x19\n\x11job_gc_timeout_ms\x18\x06 \x01(\x03\x12\x19\n\x11\x63lient_timeout_ms\x18\x08 \x01(\x03\"\xbe\x02\n\x0cWorkerConfig\x12\x0c\n\x04port\x18\x01 \x01(\x03\x12\x10\n\x08protocol\x18\x02 \x01(\t\x12\x1a\n\x12\x64ispatcher_address\x18\x03 \x01(\t\x12\x16\n\x0eworker_address\x18\x04 \x01(\t\x12\x13\n\x0bworker_tags\x18\n \x03(\t\x12\x1d\n\x15heartbeat_interval_ms\x18\x05 \x01(\x03\x12\x1d\n\x15\x64ispatcher_timeout_ms\x18\x06 \x01(\x03\x12\x1e\n\x16\x64\x61ta_transfer_protocol\x18\x07 \x01(\t\x12\x1d\n\x15\x64\x61ta_transfer_address\x18\x08 \x01(\t\x12&\n\x1e\x63ross_trainer_cache_size_bytes\x18\x0b \x01(\x03\x12 \n\x18shutdown_quiet_period_ms\x18\t \x01(\x03\x42WZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_protob\x06proto3')
  ,
  dependencies=[tensorflow_dot_core_dot_protobuf_dot_data__service__pb2.DESCRIPTOR,])




_DISPATCHERCONFIG = _descriptor.Descriptor(
  name='DispatcherConfig',
  full_name='tensorflow.data.experimental.DispatcherConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='port', full_name='tensorflow.data.experimental.DispatcherConfig.port', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='protocol', full_name='tensorflow.data.experimental.DispatcherConfig.protocol', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='work_dir', full_name='tensorflow.data.experimental.DispatcherConfig.work_dir', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fault_tolerant_mode', full_name='tensorflow.data.experimental.DispatcherConfig.fault_tolerant_mode', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='worker_addresses', full_name='tensorflow.data.experimental.DispatcherConfig.worker_addresses', index=4,
      number=7, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='deployment_mode', full_name='tensorflow.data.experimental.DispatcherConfig.deployment_mode', index=5,
      number=9, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='job_gc_check_interval_ms', full_name='tensorflow.data.experimental.DispatcherConfig.job_gc_check_interval_ms', index=6,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='job_gc_timeout_ms', full_name='tensorflow.data.experimental.DispatcherConfig.job_gc_timeout_ms', index=7,
      number=6, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='client_timeout_ms', full_name='tensorflow.data.experimental.DispatcherConfig.client_timeout_ms', index=8,
      number=8, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=125,
  serialized_end=394,
)


_WORKERCONFIG = _descriptor.Descriptor(
  name='WorkerConfig',
  full_name='tensorflow.data.experimental.WorkerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='port', full_name='tensorflow.data.experimental.WorkerConfig.port', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='protocol', full_name='tensorflow.data.experimental.WorkerConfig.protocol', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dispatcher_address', full_name='tensorflow.data.experimental.WorkerConfig.dispatcher_address', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='worker_address', full_name='tensorflow.data.experimental.WorkerConfig.worker_address', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='worker_tags', full_name='tensorflow.data.experimental.WorkerConfig.worker_tags', index=4,
      number=10, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='heartbeat_interval_ms', full_name='tensorflow.data.experimental.WorkerConfig.heartbeat_interval_ms', index=5,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dispatcher_timeout_ms', full_name='tensorflow.data.experimental.WorkerConfig.dispatcher_timeout_ms', index=6,
      number=6, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_transfer_protocol', full_name='tensorflow.data.experimental.WorkerConfig.data_transfer_protocol', index=7,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_transfer_address', full_name='tensorflow.data.experimental.WorkerConfig.data_transfer_address', index=8,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cross_trainer_cache_size_bytes', full_name='tensorflow.data.experimental.WorkerConfig.cross_trainer_cache_size_bytes', index=9,
      number=11, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shutdown_quiet_period_ms', full_name='tensorflow.data.experimental.WorkerConfig.shutdown_quiet_period_ms', index=10,
      number=9, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=397,
  serialized_end=715,
)

_DISPATCHERCONFIG.fields_by_name['deployment_mode'].enum_type = tensorflow_dot_core_dot_protobuf_dot_data__service__pb2._DEPLOYMENTMODE
DESCRIPTOR.message_types_by_name['DispatcherConfig'] = _DISPATCHERCONFIG
DESCRIPTOR.message_types_by_name['WorkerConfig'] = _WORKERCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DispatcherConfig = _reflection.GeneratedProtocolMessageType('DispatcherConfig', (_message.Message,), {
  'DESCRIPTOR' : _DISPATCHERCONFIG,
  '__module__' : 'tensorflow.core.protobuf.service_config_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.experimental.DispatcherConfig)
  })
_sym_db.RegisterMessage(DispatcherConfig)

WorkerConfig = _reflection.GeneratedProtocolMessageType('WorkerConfig', (_message.Message,), {
  'DESCRIPTOR' : _WORKERCONFIG,
  '__module__' : 'tensorflow.core.protobuf.service_config_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.experimental.WorkerConfig)
  })
_sym_db.RegisterMessage(WorkerConfig)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
