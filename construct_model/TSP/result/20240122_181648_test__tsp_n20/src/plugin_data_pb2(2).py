# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorboard/plugins/pr_curve/plugin_data.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorboard/plugins/pr_curve/plugin_data.proto',
  package='tensorboard',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n.tensorboard/plugins/pr_curve/plugin_data.proto\x12\x0btensorboard\"<\n\x11PrCurvePluginData\x12\x0f\n\x07version\x18\x01 \x01(\x05\x12\x16\n\x0enum_thresholds\x18\x02 \x01(\rb\x06proto3')
)




_PRCURVEPLUGINDATA = _descriptor.Descriptor(
  name='PrCurvePluginData',
  full_name='tensorboard.PrCurvePluginData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='version', full_name='tensorboard.PrCurvePluginData.version', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_thresholds', full_name='tensorboard.PrCurvePluginData.num_thresholds', index=1,
      number=2, type=13, cpp_type=3, label=1,
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
  serialized_start=63,
  serialized_end=123,
)

DESCRIPTOR.message_types_by_name['PrCurvePluginData'] = _PRCURVEPLUGINDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PrCurvePluginData = _reflection.GeneratedProtocolMessageType('PrCurvePluginData', (_message.Message,), {
  'DESCRIPTOR' : _PRCURVEPLUGINDATA,
  '__module__' : 'tensorboard.plugins.pr_curve.plugin_data_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.PrCurvePluginData)
  })
_sym_db.RegisterMessage(PrCurvePluginData)


# @@protoc_insertion_point(module_scope)
