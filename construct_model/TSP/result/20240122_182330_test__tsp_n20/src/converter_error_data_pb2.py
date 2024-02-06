# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/lite/python/metrics/converter_error_data.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/lite/python/metrics/converter_error_data.proto',
  package='tflite.metrics',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n9tensorflow/lite/python/metrics/converter_error_data.proto\x12\x0etflite.metrics\"\xab\x06\n\x12\x43onverterErrorData\x12\x11\n\tcomponent\x18\x01 \x01(\t\x12\x14\n\x0csubcomponent\x18\x02 \x01(\t\x12@\n\nerror_code\x18\x03 \x01(\x0e\x32,.tflite.metrics.ConverterErrorData.ErrorCode\x12\x15\n\rerror_message\x18\x04 \x01(\t\x12=\n\x08operator\x18\x05 \x01(\x0b\x32+.tflite.metrics.ConverterErrorData.Operator\x12=\n\x08location\x18\x06 \x01(\x0b\x32+.tflite.metrics.ConverterErrorData.Location\x1a\x18\n\x08Operator\x12\x0c\n\x04name\x18\x01 \x01(\t\x1a\x39\n\x07\x46ileLoc\x12\x10\n\x08\x66ilename\x18\x01 \x01(\t\x12\x0c\n\x04line\x18\x02 \x01(\r\x12\x0e\n\x06\x63olumn\x18\x03 \x01(\r\x1aU\n\tSourceLoc\x12\x0c\n\x04name\x18\x01 \x01(\t\x12:\n\x06source\x18\x02 \x01(\x0b\x32*.tflite.metrics.ConverterErrorData.FileLoc\x1a\x85\x01\n\x08Location\x12=\n\x04type\x18\x01 \x01(\x0e\x32/.tflite.metrics.ConverterErrorData.LocationType\x12:\n\x04\x63\x61ll\x18\x02 \x03(\x0b\x32,.tflite.metrics.ConverterErrorData.SourceLoc\"\x94\x01\n\tErrorCode\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x18\n\x14\x45RROR_NEEDS_FLEX_OPS\x10\x01\x12\x1a\n\x16\x45RROR_NEEDS_CUSTOM_OPS\x10\x02\x12%\n!ERROR_UNSUPPORTED_CONTROL_FLOW_V1\x10\x03\x12\x1d\n\x18\x45RROR_GPU_NOT_COMPATIBLE\x10\xc8\x01\"J\n\x0cLocationType\x12\x0e\n\nUNKNOWNLOC\x10\x00\x12\x0b\n\x07NAMELOC\x10\x01\x12\x0f\n\x0b\x43\x41LLSITELOC\x10\x02\x12\x0c\n\x08\x46USEDLOC\x10\x03')
)



_CONVERTERERRORDATA_ERRORCODE = _descriptor.EnumDescriptor(
  name='ErrorCode',
  full_name='tflite.metrics.ConverterErrorData.ErrorCode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR_NEEDS_FLEX_OPS', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR_NEEDS_CUSTOM_OPS', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR_UNSUPPORTED_CONTROL_FLOW_V1', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR_GPU_NOT_COMPATIBLE', index=4, number=200,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=665,
  serialized_end=813,
)
_sym_db.RegisterEnumDescriptor(_CONVERTERERRORDATA_ERRORCODE)

_CONVERTERERRORDATA_LOCATIONTYPE = _descriptor.EnumDescriptor(
  name='LocationType',
  full_name='tflite.metrics.ConverterErrorData.LocationType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWNLOC', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NAMELOC', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CALLSITELOC', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FUSEDLOC', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=815,
  serialized_end=889,
)
_sym_db.RegisterEnumDescriptor(_CONVERTERERRORDATA_LOCATIONTYPE)


_CONVERTERERRORDATA_OPERATOR = _descriptor.Descriptor(
  name='Operator',
  full_name='tflite.metrics.ConverterErrorData.Operator',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='tflite.metrics.ConverterErrorData.Operator.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=356,
  serialized_end=380,
)

_CONVERTERERRORDATA_FILELOC = _descriptor.Descriptor(
  name='FileLoc',
  full_name='tflite.metrics.ConverterErrorData.FileLoc',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filename', full_name='tflite.metrics.ConverterErrorData.FileLoc.filename', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='line', full_name='tflite.metrics.ConverterErrorData.FileLoc.line', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='column', full_name='tflite.metrics.ConverterErrorData.FileLoc.column', index=2,
      number=3, type=13, cpp_type=3, label=1,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=382,
  serialized_end=439,
)

_CONVERTERERRORDATA_SOURCELOC = _descriptor.Descriptor(
  name='SourceLoc',
  full_name='tflite.metrics.ConverterErrorData.SourceLoc',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='tflite.metrics.ConverterErrorData.SourceLoc.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='source', full_name='tflite.metrics.ConverterErrorData.SourceLoc.source', index=1,
      number=2, type=11, cpp_type=10, label=1,
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
  serialized_start=441,
  serialized_end=526,
)

_CONVERTERERRORDATA_LOCATION = _descriptor.Descriptor(
  name='Location',
  full_name='tflite.metrics.ConverterErrorData.Location',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='tflite.metrics.ConverterErrorData.Location.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='call', full_name='tflite.metrics.ConverterErrorData.Location.call', index=1,
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
  serialized_start=529,
  serialized_end=662,
)

_CONVERTERERRORDATA = _descriptor.Descriptor(
  name='ConverterErrorData',
  full_name='tflite.metrics.ConverterErrorData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='component', full_name='tflite.metrics.ConverterErrorData.component', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='subcomponent', full_name='tflite.metrics.ConverterErrorData.subcomponent', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='error_code', full_name='tflite.metrics.ConverterErrorData.error_code', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='error_message', full_name='tflite.metrics.ConverterErrorData.error_message', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='operator', full_name='tflite.metrics.ConverterErrorData.operator', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='location', full_name='tflite.metrics.ConverterErrorData.location', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_CONVERTERERRORDATA_OPERATOR, _CONVERTERERRORDATA_FILELOC, _CONVERTERERRORDATA_SOURCELOC, _CONVERTERERRORDATA_LOCATION, ],
  enum_types=[
    _CONVERTERERRORDATA_ERRORCODE,
    _CONVERTERERRORDATA_LOCATIONTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=78,
  serialized_end=889,
)

_CONVERTERERRORDATA_OPERATOR.containing_type = _CONVERTERERRORDATA
_CONVERTERERRORDATA_FILELOC.containing_type = _CONVERTERERRORDATA
_CONVERTERERRORDATA_SOURCELOC.fields_by_name['source'].message_type = _CONVERTERERRORDATA_FILELOC
_CONVERTERERRORDATA_SOURCELOC.containing_type = _CONVERTERERRORDATA
_CONVERTERERRORDATA_LOCATION.fields_by_name['type'].enum_type = _CONVERTERERRORDATA_LOCATIONTYPE
_CONVERTERERRORDATA_LOCATION.fields_by_name['call'].message_type = _CONVERTERERRORDATA_SOURCELOC
_CONVERTERERRORDATA_LOCATION.containing_type = _CONVERTERERRORDATA
_CONVERTERERRORDATA.fields_by_name['error_code'].enum_type = _CONVERTERERRORDATA_ERRORCODE
_CONVERTERERRORDATA.fields_by_name['operator'].message_type = _CONVERTERERRORDATA_OPERATOR
_CONVERTERERRORDATA.fields_by_name['location'].message_type = _CONVERTERERRORDATA_LOCATION
_CONVERTERERRORDATA_ERRORCODE.containing_type = _CONVERTERERRORDATA
_CONVERTERERRORDATA_LOCATIONTYPE.containing_type = _CONVERTERERRORDATA
DESCRIPTOR.message_types_by_name['ConverterErrorData'] = _CONVERTERERRORDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ConverterErrorData = _reflection.GeneratedProtocolMessageType('ConverterErrorData', (_message.Message,), {

  'Operator' : _reflection.GeneratedProtocolMessageType('Operator', (_message.Message,), {
    'DESCRIPTOR' : _CONVERTERERRORDATA_OPERATOR,
    '__module__' : 'tensorflow.lite.python.metrics.converter_error_data_pb2'
    # @@protoc_insertion_point(class_scope:tflite.metrics.ConverterErrorData.Operator)
    })
  ,

  'FileLoc' : _reflection.GeneratedProtocolMessageType('FileLoc', (_message.Message,), {
    'DESCRIPTOR' : _CONVERTERERRORDATA_FILELOC,
    '__module__' : 'tensorflow.lite.python.metrics.converter_error_data_pb2'
    # @@protoc_insertion_point(class_scope:tflite.metrics.ConverterErrorData.FileLoc)
    })
  ,

  'SourceLoc' : _reflection.GeneratedProtocolMessageType('SourceLoc', (_message.Message,), {
    'DESCRIPTOR' : _CONVERTERERRORDATA_SOURCELOC,
    '__module__' : 'tensorflow.lite.python.metrics.converter_error_data_pb2'
    # @@protoc_insertion_point(class_scope:tflite.metrics.ConverterErrorData.SourceLoc)
    })
  ,

  'Location' : _reflection.GeneratedProtocolMessageType('Location', (_message.Message,), {
    'DESCRIPTOR' : _CONVERTERERRORDATA_LOCATION,
    '__module__' : 'tensorflow.lite.python.metrics.converter_error_data_pb2'
    # @@protoc_insertion_point(class_scope:tflite.metrics.ConverterErrorData.Location)
    })
  ,
  'DESCRIPTOR' : _CONVERTERERRORDATA,
  '__module__' : 'tensorflow.lite.python.metrics.converter_error_data_pb2'
  # @@protoc_insertion_point(class_scope:tflite.metrics.ConverterErrorData)
  })
_sym_db.RegisterMessage(ConverterErrorData)
_sym_db.RegisterMessage(ConverterErrorData.Operator)
_sym_db.RegisterMessage(ConverterErrorData.FileLoc)
_sym_db.RegisterMessage(ConverterErrorData.SourceLoc)
_sym_db.RegisterMessage(ConverterErrorData.Location)


# @@protoc_insertion_point(module_scope)
