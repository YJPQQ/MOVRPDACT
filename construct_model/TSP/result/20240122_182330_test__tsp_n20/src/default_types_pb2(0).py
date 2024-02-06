# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/function/trace_type/default_types.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow.core.function.trace_type import serialization_pb2 as tensorflow_dot_core_dot_function_dot_trace__type_dot_serialization__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/core/function/trace_type/default_types.proto',
  package='tensorflow.core.function.trace_type.default_types',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n7tensorflow/core/function/trace_type/default_types.proto\x12\x31tensorflow.core.function.trace_type.default_types\x1a\x37tensorflow/core/function/trace_type/serialization.proto\"\xe6\x01\n\x11SerializedLiteral\x12\x14\n\nbool_value\x18\x01 \x01(\x08H\x00\x12\x13\n\tint_value\x18\x02 \x01(\x03H\x00\x12\x15\n\x0b\x66loat_value\x18\x03 \x01(\x01H\x00\x12\x13\n\tstr_value\x18\x04 \x01(\tH\x00\x12\x64\n\nnone_value\x18\x05 \x01(\x0b\x32N.tensorflow.core.function.trace_type.default_types.SerializedLiteral.NoneValueH\x00\x1a\x0b\n\tNoneValueB\x07\n\x05value\"m\n\x0fSerializedTuple\x12Z\n\ncomponents\x18\x01 \x03(\x0b\x32\x46.tensorflow.core.function.trace_type.serialization.SerializedTraceType\"n\n\x0eSerializedList\x12\\\n\x10\x63omponents_tuple\x18\x01 \x01(\x0b\x32\x42.tensorflow.core.function.trace_type.default_types.SerializedTuple\"\x9a\x01\n\x14SerializedNamedTuple\x12\x11\n\ttype_name\x18\x01 \x01(\t\x12\x17\n\x0f\x61ttribute_names\x18\x02 \x03(\t\x12V\n\nattributes\x18\x03 \x01(\x0b\x32\x42.tensorflow.core.function.trace_type.default_types.SerializedTuple\"t\n\x0fSerializedAttrs\x12\x61\n\x10named_attributes\x18\x01 \x01(\x0b\x32G.tensorflow.core.function.trace_type.default_types.SerializedNamedTuple\"\xbc\x01\n\x0eSerializedDict\x12R\n\x04keys\x18\x01 \x03(\x0b\x32\x44.tensorflow.core.function.trace_type.default_types.SerializedLiteral\x12V\n\x06values\x18\x02 \x03(\x0b\x32\x46.tensorflow.core.function.trace_type.serialization.SerializedTraceType')
  ,
  dependencies=[tensorflow_dot_core_dot_function_dot_trace__type_dot_serialization__pb2.DESCRIPTOR,])




_SERIALIZEDLITERAL_NONEVALUE = _descriptor.Descriptor(
  name='NoneValue',
  full_name='tensorflow.core.function.trace_type.default_types.SerializedLiteral.NoneValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=378,
  serialized_end=389,
)

_SERIALIZEDLITERAL = _descriptor.Descriptor(
  name='SerializedLiteral',
  full_name='tensorflow.core.function.trace_type.default_types.SerializedLiteral',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='bool_value', full_name='tensorflow.core.function.trace_type.default_types.SerializedLiteral.bool_value', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='int_value', full_name='tensorflow.core.function.trace_type.default_types.SerializedLiteral.int_value', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='float_value', full_name='tensorflow.core.function.trace_type.default_types.SerializedLiteral.float_value', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='str_value', full_name='tensorflow.core.function.trace_type.default_types.SerializedLiteral.str_value', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='none_value', full_name='tensorflow.core.function.trace_type.default_types.SerializedLiteral.none_value', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_SERIALIZEDLITERAL_NONEVALUE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='value', full_name='tensorflow.core.function.trace_type.default_types.SerializedLiteral.value',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=168,
  serialized_end=398,
)


_SERIALIZEDTUPLE = _descriptor.Descriptor(
  name='SerializedTuple',
  full_name='tensorflow.core.function.trace_type.default_types.SerializedTuple',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='components', full_name='tensorflow.core.function.trace_type.default_types.SerializedTuple.components', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=400,
  serialized_end=509,
)


_SERIALIZEDLIST = _descriptor.Descriptor(
  name='SerializedList',
  full_name='tensorflow.core.function.trace_type.default_types.SerializedList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='components_tuple', full_name='tensorflow.core.function.trace_type.default_types.SerializedList.components_tuple', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=511,
  serialized_end=621,
)


_SERIALIZEDNAMEDTUPLE = _descriptor.Descriptor(
  name='SerializedNamedTuple',
  full_name='tensorflow.core.function.trace_type.default_types.SerializedNamedTuple',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type_name', full_name='tensorflow.core.function.trace_type.default_types.SerializedNamedTuple.type_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attribute_names', full_name='tensorflow.core.function.trace_type.default_types.SerializedNamedTuple.attribute_names', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attributes', full_name='tensorflow.core.function.trace_type.default_types.SerializedNamedTuple.attributes', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=624,
  serialized_end=778,
)


_SERIALIZEDATTRS = _descriptor.Descriptor(
  name='SerializedAttrs',
  full_name='tensorflow.core.function.trace_type.default_types.SerializedAttrs',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='named_attributes', full_name='tensorflow.core.function.trace_type.default_types.SerializedAttrs.named_attributes', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=780,
  serialized_end=896,
)


_SERIALIZEDDICT = _descriptor.Descriptor(
  name='SerializedDict',
  full_name='tensorflow.core.function.trace_type.default_types.SerializedDict',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='keys', full_name='tensorflow.core.function.trace_type.default_types.SerializedDict.keys', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='values', full_name='tensorflow.core.function.trace_type.default_types.SerializedDict.values', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=899,
  serialized_end=1087,
)

_SERIALIZEDLITERAL_NONEVALUE.containing_type = _SERIALIZEDLITERAL
_SERIALIZEDLITERAL.fields_by_name['none_value'].message_type = _SERIALIZEDLITERAL_NONEVALUE
_SERIALIZEDLITERAL.oneofs_by_name['value'].fields.append(
  _SERIALIZEDLITERAL.fields_by_name['bool_value'])
_SERIALIZEDLITERAL.fields_by_name['bool_value'].containing_oneof = _SERIALIZEDLITERAL.oneofs_by_name['value']
_SERIALIZEDLITERAL.oneofs_by_name['value'].fields.append(
  _SERIALIZEDLITERAL.fields_by_name['int_value'])
_SERIALIZEDLITERAL.fields_by_name['int_value'].containing_oneof = _SERIALIZEDLITERAL.oneofs_by_name['value']
_SERIALIZEDLITERAL.oneofs_by_name['value'].fields.append(
  _SERIALIZEDLITERAL.fields_by_name['float_value'])
_SERIALIZEDLITERAL.fields_by_name['float_value'].containing_oneof = _SERIALIZEDLITERAL.oneofs_by_name['value']
_SERIALIZEDLITERAL.oneofs_by_name['value'].fields.append(
  _SERIALIZEDLITERAL.fields_by_name['str_value'])
_SERIALIZEDLITERAL.fields_by_name['str_value'].containing_oneof = _SERIALIZEDLITERAL.oneofs_by_name['value']
_SERIALIZEDLITERAL.oneofs_by_name['value'].fields.append(
  _SERIALIZEDLITERAL.fields_by_name['none_value'])
_SERIALIZEDLITERAL.fields_by_name['none_value'].containing_oneof = _SERIALIZEDLITERAL.oneofs_by_name['value']
_SERIALIZEDTUPLE.fields_by_name['components'].message_type = tensorflow_dot_core_dot_function_dot_trace__type_dot_serialization__pb2._SERIALIZEDTRACETYPE
_SERIALIZEDLIST.fields_by_name['components_tuple'].message_type = _SERIALIZEDTUPLE
_SERIALIZEDNAMEDTUPLE.fields_by_name['attributes'].message_type = _SERIALIZEDTUPLE
_SERIALIZEDATTRS.fields_by_name['named_attributes'].message_type = _SERIALIZEDNAMEDTUPLE
_SERIALIZEDDICT.fields_by_name['keys'].message_type = _SERIALIZEDLITERAL
_SERIALIZEDDICT.fields_by_name['values'].message_type = tensorflow_dot_core_dot_function_dot_trace__type_dot_serialization__pb2._SERIALIZEDTRACETYPE
DESCRIPTOR.message_types_by_name['SerializedLiteral'] = _SERIALIZEDLITERAL
DESCRIPTOR.message_types_by_name['SerializedTuple'] = _SERIALIZEDTUPLE
DESCRIPTOR.message_types_by_name['SerializedList'] = _SERIALIZEDLIST
DESCRIPTOR.message_types_by_name['SerializedNamedTuple'] = _SERIALIZEDNAMEDTUPLE
DESCRIPTOR.message_types_by_name['SerializedAttrs'] = _SERIALIZEDATTRS
DESCRIPTOR.message_types_by_name['SerializedDict'] = _SERIALIZEDDICT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SerializedLiteral = _reflection.GeneratedProtocolMessageType('SerializedLiteral', (_message.Message,), {

  'NoneValue' : _reflection.GeneratedProtocolMessageType('NoneValue', (_message.Message,), {
    'DESCRIPTOR' : _SERIALIZEDLITERAL_NONEVALUE,
    '__module__' : 'tensorflow.core.function.trace_type.default_types_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.core.function.trace_type.default_types.SerializedLiteral.NoneValue)
    })
  ,
  'DESCRIPTOR' : _SERIALIZEDLITERAL,
  '__module__' : 'tensorflow.core.function.trace_type.default_types_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.core.function.trace_type.default_types.SerializedLiteral)
  })
_sym_db.RegisterMessage(SerializedLiteral)
_sym_db.RegisterMessage(SerializedLiteral.NoneValue)

SerializedTuple = _reflection.GeneratedProtocolMessageType('SerializedTuple', (_message.Message,), {
  'DESCRIPTOR' : _SERIALIZEDTUPLE,
  '__module__' : 'tensorflow.core.function.trace_type.default_types_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.core.function.trace_type.default_types.SerializedTuple)
  })
_sym_db.RegisterMessage(SerializedTuple)

SerializedList = _reflection.GeneratedProtocolMessageType('SerializedList', (_message.Message,), {
  'DESCRIPTOR' : _SERIALIZEDLIST,
  '__module__' : 'tensorflow.core.function.trace_type.default_types_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.core.function.trace_type.default_types.SerializedList)
  })
_sym_db.RegisterMessage(SerializedList)

SerializedNamedTuple = _reflection.GeneratedProtocolMessageType('SerializedNamedTuple', (_message.Message,), {
  'DESCRIPTOR' : _SERIALIZEDNAMEDTUPLE,
  '__module__' : 'tensorflow.core.function.trace_type.default_types_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.core.function.trace_type.default_types.SerializedNamedTuple)
  })
_sym_db.RegisterMessage(SerializedNamedTuple)

SerializedAttrs = _reflection.GeneratedProtocolMessageType('SerializedAttrs', (_message.Message,), {
  'DESCRIPTOR' : _SERIALIZEDATTRS,
  '__module__' : 'tensorflow.core.function.trace_type.default_types_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.core.function.trace_type.default_types.SerializedAttrs)
  })
_sym_db.RegisterMessage(SerializedAttrs)

SerializedDict = _reflection.GeneratedProtocolMessageType('SerializedDict', (_message.Message,), {
  'DESCRIPTOR' : _SERIALIZEDDICT,
  '__module__' : 'tensorflow.core.function.trace_type.default_types_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.core.function.trace_type.default_types.SerializedDict)
  })
_sym_db.RegisterMessage(SerializedDict)


# @@protoc_insertion_point(module_scope)
