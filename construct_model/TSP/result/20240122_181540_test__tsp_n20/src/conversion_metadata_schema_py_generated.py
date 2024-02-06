import flatbuffers

# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

from flatbuffers.compat import import_numpy
np = import_numpy()

class ModelType(object):
    NONE = 0
    TF_SAVED_MODEL = 1
    KERAS_MODEL = 2
    TF_CONCRETE_FUNCTIONS = 3
    TF_GRAPH_DEF = 4
    TF_SESSION = 5
    JAX = 6


class ModelOptimizationMode(object):
    PTQ_FLOAT16 = 1001
    PTQ_DYNAMIC_RANGE = 1002
    PTQ_FULL_INTEGER = 1003
    PTQ_INT16 = 1004
    QUANTIZATION_AWARE_TRAINING = 2000
    RANDOM_SPARSITY = 3001
    BLOCK_SPARSITY = 3002
    STRUCTURED_SPARSITY = 3003


class Environment(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Environment()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsEnvironment(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Environment
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Environment
    def TensorflowVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Environment
    def ApiVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Environment
    def ModelType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def EnvironmentStart(builder): builder.StartObject(3)
def EnvironmentAddTensorflowVersion(builder, tensorflowVersion): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(tensorflowVersion), 0)
def EnvironmentAddApiVersion(builder, apiVersion): builder.PrependUint32Slot(1, apiVersion, 0)
def EnvironmentAddModelType(builder, modelType): builder.PrependInt32Slot(2, modelType, 0)
def EnvironmentEnd(builder): return builder.EndObject()


class EnvironmentT(object):

    # EnvironmentT
    def __init__(self):
        self.tensorflowVersion = None  # type: str
        self.apiVersion = 0  # type: int
        self.modelType = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        environment = Environment()
        environment.Init(buf, pos)
        return cls.InitFromObj(environment)

    @classmethod
    def InitFromObj(cls, environment):
        x = EnvironmentT()
        x._UnPack(environment)
        return x

    # EnvironmentT
    def _UnPack(self, environment):
        if environment is None:
            return
        self.tensorflowVersion = environment.TensorflowVersion()
        self.apiVersion = environment.ApiVersion()
        self.modelType = environment.ModelType()

    # EnvironmentT
    def Pack(self, builder):
        if self.tensorflowVersion is not None:
            tensorflowVersion = builder.CreateString(self.tensorflowVersion)
        EnvironmentStart(builder)
        if self.tensorflowVersion is not None:
            EnvironmentAddTensorflowVersion(builder, tensorflowVersion)
        EnvironmentAddApiVersion(builder, self.apiVersion)
        EnvironmentAddModelType(builder, self.modelType)
        environment = EnvironmentEnd(builder)
        return environment


class SparsityBlockSize(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SparsityBlockSize()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSparsityBlockSize(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # SparsityBlockSize
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SparsityBlockSize
    def Values(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SparsityBlockSize
    def ValuesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint32Flags, o)
        return 0

    # SparsityBlockSize
    def ValuesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SparsityBlockSize
    def ValuesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def SparsityBlockSizeStart(builder): builder.StartObject(1)
def SparsityBlockSizeAddValues(builder, values): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(values), 0)
def SparsityBlockSizeStartValuesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SparsityBlockSizeEnd(builder): return builder.EndObject()

try:
    from typing import List
except:
    pass

class SparsityBlockSizeT(object):

    # SparsityBlockSizeT
    def __init__(self):
        self.values = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sparsityBlockSize = SparsityBlockSize()
        sparsityBlockSize.Init(buf, pos)
        return cls.InitFromObj(sparsityBlockSize)

    @classmethod
    def InitFromObj(cls, sparsityBlockSize):
        x = SparsityBlockSizeT()
        x._UnPack(sparsityBlockSize)
        return x

    # SparsityBlockSizeT
    def _UnPack(self, sparsityBlockSize):
        if sparsityBlockSize is None:
            return
        if not sparsityBlockSize.ValuesIsNone():
            if np is None:
                self.values = []
                for i in range(sparsityBlockSize.ValuesLength()):
                    self.values.append(sparsityBlockSize.Values(i))
            else:
                self.values = sparsityBlockSize.ValuesAsNumpy()

    # SparsityBlockSizeT
    def Pack(self, builder):
        if self.values is not None:
            if np is not None and type(self.values) is np.ndarray:
                values = builder.CreateNumpyVector(self.values)
            else:
                SparsityBlockSizeStartValuesVector(builder, len(self.values))
                for i in reversed(range(len(self.values))):
                    builder.PrependUint32(self.values[i])
                values = builder.EndVector()
        SparsityBlockSizeStart(builder)
        if self.values is not None:
            SparsityBlockSizeAddValues(builder, values)
        sparsityBlockSize = SparsityBlockSizeEnd(builder)
        return sparsityBlockSize


class ConversionOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ConversionOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsConversionOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # ConversionOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ConversionOptions
    def ModelOptimizationModes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # ConversionOptions
    def ModelOptimizationModesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ConversionOptions
    def ModelOptimizationModesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ConversionOptions
    def ModelOptimizationModesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # ConversionOptions
    def AllowCustomOps(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # ConversionOptions
    def EnableSelectTfOps(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # ConversionOptions
    def ForceSelectTfOps(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # ConversionOptions
    def SparsityBlockSizes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = SparsityBlockSize()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ConversionOptions
    def SparsityBlockSizesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ConversionOptions
    def SparsityBlockSizesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

def ConversionOptionsStart(builder): builder.StartObject(5)
def ConversionOptionsAddModelOptimizationModes(builder, modelOptimizationModes): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(modelOptimizationModes), 0)
def ConversionOptionsStartModelOptimizationModesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ConversionOptionsAddAllowCustomOps(builder, allowCustomOps): builder.PrependBoolSlot(1, allowCustomOps, 0)
def ConversionOptionsAddEnableSelectTfOps(builder, enableSelectTfOps): builder.PrependBoolSlot(2, enableSelectTfOps, 0)
def ConversionOptionsAddForceSelectTfOps(builder, forceSelectTfOps): builder.PrependBoolSlot(3, forceSelectTfOps, 0)
def ConversionOptionsAddSparsityBlockSizes(builder, sparsityBlockSizes): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(sparsityBlockSizes), 0)
def ConversionOptionsStartSparsityBlockSizesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ConversionOptionsEnd(builder): return builder.EndObject()

try:
    from typing import List
except:
    pass

class ConversionOptionsT(object):

    # ConversionOptionsT
    def __init__(self):
        self.modelOptimizationModes = None  # type: List[int]
        self.allowCustomOps = False  # type: bool
        self.enableSelectTfOps = False  # type: bool
        self.forceSelectTfOps = False  # type: bool
        self.sparsityBlockSizes = None  # type: List[SparsityBlockSizeT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        conversionOptions = ConversionOptions()
        conversionOptions.Init(buf, pos)
        return cls.InitFromObj(conversionOptions)

    @classmethod
    def InitFromObj(cls, conversionOptions):
        x = ConversionOptionsT()
        x._UnPack(conversionOptions)
        return x

    # ConversionOptionsT
    def _UnPack(self, conversionOptions):
        if conversionOptions is None:
            return
        if not conversionOptions.ModelOptimizationModesIsNone():
            if np is None:
                self.modelOptimizationModes = []
                for i in range(conversionOptions.ModelOptimizationModesLength()):
                    self.modelOptimizationModes.append(conversionOptions.ModelOptimizationModes(i))
            else:
                self.modelOptimizationModes = conversionOptions.ModelOptimizationModesAsNumpy()
        self.allowCustomOps = conversionOptions.AllowCustomOps()
        self.enableSelectTfOps = conversionOptions.EnableSelectTfOps()
        self.forceSelectTfOps = conversionOptions.ForceSelectTfOps()
        if not conversionOptions.SparsityBlockSizesIsNone():
            self.sparsityBlockSizes = []
            for i in range(conversionOptions.SparsityBlockSizesLength()):
                if conversionOptions.SparsityBlockSizes(i) is None:
                    self.sparsityBlockSizes.append(None)
                else:
                    sparsityBlockSize_ = SparsityBlockSizeT.InitFromObj(conversionOptions.SparsityBlockSizes(i))
                    self.sparsityBlockSizes.append(sparsityBlockSize_)

    # ConversionOptionsT
    def Pack(self, builder):
        if self.modelOptimizationModes is not None:
            if np is not None and type(self.modelOptimizationModes) is np.ndarray:
                modelOptimizationModes = builder.CreateNumpyVector(self.modelOptimizationModes)
            else:
                ConversionOptionsStartModelOptimizationModesVector(builder, len(self.modelOptimizationModes))
                for i in reversed(range(len(self.modelOptimizationModes))):
                    builder.PrependInt32(self.modelOptimizationModes[i])
                modelOptimizationModes = builder.EndVector()
        if self.sparsityBlockSizes is not None:
            sparsityBlockSizeslist = []
            for i in range(len(self.sparsityBlockSizes)):
                sparsityBlockSizeslist.append(self.sparsityBlockSizes[i].Pack(builder))
            ConversionOptionsStartSparsityBlockSizesVector(builder, len(self.sparsityBlockSizes))
            for i in reversed(range(len(self.sparsityBlockSizes))):
                builder.PrependUOffsetTRelative(sparsityBlockSizeslist[i])
            sparsityBlockSizes = builder.EndVector()
        ConversionOptionsStart(builder)
        if self.modelOptimizationModes is not None:
            ConversionOptionsAddModelOptimizationModes(builder, modelOptimizationModes)
        ConversionOptionsAddAllowCustomOps(builder, self.allowCustomOps)
        ConversionOptionsAddEnableSelectTfOps(builder, self.enableSelectTfOps)
        ConversionOptionsAddForceSelectTfOps(builder, self.forceSelectTfOps)
        if self.sparsityBlockSizes is not None:
            ConversionOptionsAddSparsityBlockSizes(builder, sparsityBlockSizes)
        conversionOptions = ConversionOptionsEnd(builder)
        return conversionOptions


class ConversionMetadata(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ConversionMetadata()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsConversionMetadata(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # ConversionMetadata
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ConversionMetadata
    def Environment(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = Environment()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ConversionMetadata
    def Options(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = ConversionOptions()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def ConversionMetadataStart(builder): builder.StartObject(2)
def ConversionMetadataAddEnvironment(builder, environment): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(environment), 0)
def ConversionMetadataAddOptions(builder, options): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(options), 0)
def ConversionMetadataEnd(builder): return builder.EndObject()

try:
    from typing import Optional
except:
    pass

class ConversionMetadataT(object):

    # ConversionMetadataT
    def __init__(self):
        self.environment = None  # type: Optional[EnvironmentT]
        self.options = None  # type: Optional[ConversionOptionsT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        conversionMetadata = ConversionMetadata()
        conversionMetadata.Init(buf, pos)
        return cls.InitFromObj(conversionMetadata)

    @classmethod
    def InitFromObj(cls, conversionMetadata):
        x = ConversionMetadataT()
        x._UnPack(conversionMetadata)
        return x

    # ConversionMetadataT
    def _UnPack(self, conversionMetadata):
        if conversionMetadata is None:
            return
        if conversionMetadata.Environment() is not None:
            self.environment = EnvironmentT.InitFromObj(conversionMetadata.Environment())
        if conversionMetadata.Options() is not None:
            self.options = ConversionOptionsT.InitFromObj(conversionMetadata.Options())

    # ConversionMetadataT
    def Pack(self, builder):
        if self.environment is not None:
            environment = self.environment.Pack(builder)
        if self.options is not None:
            options = self.options.Pack(builder)
        ConversionMetadataStart(builder)
        if self.environment is not None:
            ConversionMetadataAddEnvironment(builder, environment)
        if self.options is not None:
            ConversionMetadataAddOptions(builder, options)
        conversionMetadata = ConversionMetadataEnd(builder)
        return conversionMetadata


