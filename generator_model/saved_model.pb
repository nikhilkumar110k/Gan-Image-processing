¤Г
ъ║
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
└
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resourceИ
√
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%═╠L>"
Ttype0:
2
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
е
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeКэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
░
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48Се
░
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *(

debug_nameconv2d_transpose_2/bias/*
dtype0*
shape:*(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
:*
dtype0
┬
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: **

debug_nameconv2d_transpose_2/kernel/*
dtype0*
shape:@**
shared_nameconv2d_transpose_2/kernel
П
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
:@*
dtype0
┌
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_2/moving_variance/*
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
╬
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_2/moving_mean/*
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
╣
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_2/beta/*
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
╝
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_2/gamma/*
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
░
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *(

debug_nameconv2d_transpose_1/bias/*
dtype0*
shape:@*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:@*
dtype0
├
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: **

debug_nameconv2d_transpose_1/kernel/*
dtype0*
shape:@А**
shared_nameconv2d_transpose_1/kernel
Р
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*'
_output_shapes
:@А*
dtype0
█
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_1/moving_variance/*
dtype0*
shape:А*6
shared_name'%batch_normalization_1/moving_variance
Ь
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:А*
dtype0
╧
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_1/moving_mean/*
dtype0*
shape:А*2
shared_name#!batch_normalization_1/moving_mean
Ф
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:А*
dtype0
║
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_1/beta/*
dtype0*
shape:А*+
shared_namebatch_normalization_1/beta
Ж
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:А*
dtype0
╜
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_1/gamma/*
dtype0*
shape:А*,
shared_namebatch_normalization_1/gamma
И
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:А*
dtype0
л
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *&

debug_nameconv2d_transpose/bias/*
dtype0*
shape:А*&
shared_nameconv2d_transpose/bias
|
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes	
:А*
dtype0
╛
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *(

debug_nameconv2d_transpose/kernel/*
dtype0*
shape:АА*(
shared_nameconv2d_transpose/kernel
Н
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*(
_output_shapes
:АА*
dtype0
╒
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *4

debug_name&$batch_normalization/moving_variance/*
dtype0*
shape:А1*4
shared_name%#batch_normalization/moving_variance
Ш
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:А1*
dtype0
╔
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *0

debug_name" batch_normalization/moving_mean/*
dtype0*
shape:А1*0
shared_name!batch_normalization/moving_mean
Р
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:А1*
dtype0
┤
batch_normalization/betaVarHandleOp*
_output_shapes
: *)

debug_namebatch_normalization/beta/*
dtype0*
shape:А1*)
shared_namebatch_normalization/beta
В
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:А1*
dtype0
╖
batch_normalization/gammaVarHandleOp*
_output_shapes
: **

debug_namebatch_normalization/gamma/*
dtype0*
shape:А1**
shared_namebatch_normalization/gamma
Д
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:А1*
dtype0
Р
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape:А1*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:А1*
dtype0
Ъ
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape:	dА1*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	dА1*
dtype0
▒
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *'

debug_nameembedding_2/embeddings/*
dtype0*
shape
:
d*'
shared_nameembedding_2/embeddings
Б
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*
_output_shapes

:
d*
dtype0
z
serving_default_input_3Placeholder*'
_output_shapes
:         d*
dtype0*
shape:         d
z
serving_default_input_4Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
╬
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3serving_default_input_4embedding_2/embeddingsdense_1/kerneldense_1/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaconv2d_transpose/kernelconv2d_transpose/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_transpose_1/kernelconv2d_transpose_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_transpose_2/kernelconv2d_transpose_2/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_2926349

NoOpNoOp
╪\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*У\
valueЙ\BЖ\ B [
у
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
а
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
* 
О
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
О
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 
ж
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias*
О
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses* 
╒
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance*
О
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 
╚
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias
 S_jit_compiled_convolution_op*
О
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses* 
╒
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
`axis
	agamma
bbeta
cmoving_mean
dmoving_variance*
╚
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias
 m_jit_compiled_convolution_op*
О
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 
╒
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses
zaxis
	{gamma
|beta
}moving_mean
~moving_variance*
╨
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses
Еkernel
	Жbias
!З_jit_compiled_convolution_op*
д
0
21
32
A3
B4
C5
D6
Q7
R8
a9
b10
c11
d12
k13
l14
{15
|16
}17
~18
Е19
Ж20*
t
0
21
32
A3
B4
Q5
R6
a7
b8
k9
l10
{11
|12
Е13
Ж14*
* 
╡
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Нtrace_0
Оtrace_1* 

Пtrace_0
Рtrace_1* 
* 

Сserving_default* 

0*

0*
* 
Ш
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Чtrace_0* 

Шtrace_0* 
jd
VARIABLE_VALUEembedding_2/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

Юtrace_0* 

Яtrace_0* 
* 
* 
* 
Ц
аnon_trainable_variables
бlayers
вmetrics
 гlayer_regularization_losses
дlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 

еtrace_0* 

жtrace_0* 

20
31*

20
31*
* 
Ш
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

мtrace_0* 

нtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
оnon_trainable_variables
пlayers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 

│trace_0* 

┤trace_0* 
 
A0
B1
C2
D3*

A0
B1*
* 
Ш
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

║trace_0
╗trace_1* 

╝trace_0
╜trace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 

├trace_0* 

─trace_0* 

Q0
R1*

Q0
R1*
* 
Ш
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

╩trace_0* 

╦trace_0* 
ga
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

╤trace_0* 

╥trace_0* 
 
a0
b1
c2
d3*

a0
b1*
* 
Ш
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

╪trace_0
┘trace_1* 

┌trace_0
█trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

k0
l1*

k0
l1*
* 
Ш
▄non_trainable_variables
▌layers
▐metrics
 ▀layer_regularization_losses
рlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

сtrace_0* 

тtrace_0* 
ic
VARIABLE_VALUEconv2d_transpose_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses* 

шtrace_0* 

щtrace_0* 
 
{0
|1
}2
~3*

{0
|1*
* 
Ш
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*

яtrace_0
Ёtrace_1* 

ёtrace_0
Єtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

Е0
Ж1*

Е0
Ж1*
* 
Э
єnon_trainable_variables
Їlayers
їmetrics
 Ўlayer_regularization_losses
ўlayer_metrics
	variables
Аtrainable_variables
Бregularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses*

°trace_0* 

∙trace_0* 
ic
VARIABLE_VALUEconv2d_transpose_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
C0
D1
c2
d3
}4
~5*
z
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

C0
D1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

c0
d1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

}0
~1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
А
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameembedding_2/embeddingsdense_1/kerneldense_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_transpose/kernelconv2d_transpose/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_transpose_1/kernelconv2d_transpose_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_transpose_2/kernelconv2d_transpose_2/biasConst*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_save_2926936
√
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_2/embeddingsdense_1/kerneldense_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_transpose/kernelconv2d_transpose/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_transpose_1/kernelconv2d_transpose_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_transpose_2/kernelconv2d_transpose_2/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__traced_restore_2927008Ё∙

Ы
┼
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2926612

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%═╠L>╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
У

╥
7__inference_batch_normalization_2_layer_call_fn_2926708

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2925906Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926704:'#
!
_user_specified_name	2926702:'#
!
_user_specified_name	2926700:'#
!
_user_specified_name	2926698:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ы
┼
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2925784

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%═╠L>╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ш
f
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2926024

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         А1*
alpha%
╫#<`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         А1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А1:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
ЦL
є

D__inference_model_1_layer_call_and_return_conditional_losses_2926157
input_3
input_4%
embedding_2_2926100:
d"
dense_1_2926105:	dА1
dense_1_2926107:	А1*
batch_normalization_2926111:	А1*
batch_normalization_2926113:	А1*
batch_normalization_2926115:	А1*
batch_normalization_2926117:	А14
conv2d_transpose_2926121:АА'
conv2d_transpose_2926123:	А,
batch_normalization_1_2926127:	А,
batch_normalization_1_2926129:	А,
batch_normalization_1_2926131:	А,
batch_normalization_1_2926133:	А5
conv2d_transpose_1_2926136:@А(
conv2d_transpose_1_2926138:@+
batch_normalization_2_2926142:@+
batch_normalization_2_2926144:@+
batch_normalization_2_2926146:@+
batch_normalization_2_2926148:@4
conv2d_transpose_2_2926151:@(
conv2d_transpose_2_2926153:
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв(conv2d_transpose/StatefulPartitionedCallв*conv2d_transpose_1/StatefulPartitionedCallв*conv2d_transpose_2/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв#embedding_2/StatefulPartitionedCallэ
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_4embedding_2_2926100*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_embedding_2_layer_call_and_return_conditional_losses_2925987с
flatten_3/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_2925996у
multiply_1/PartitionedCallPartitionedCallinput_3"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_multiply_1_layer_call_and_return_conditional_losses_2926003Н
dense_1/StatefulPartitionedCallStatefulPartitionedCall#multiply_1/PartitionedCall:output:0dense_1_2926105dense_1_2926107*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2926014ц
leaky_re_lu_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2926024■
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2926111batch_normalization_2926113batch_normalization_2926115batch_normalization_2926117*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2925698Є
reshape_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_2926048╕
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_2926121conv2d_transpose_2926123*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2925757ў
leaky_re_lu_3/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2926059Т
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_1_2926127batch_normalization_1_2926129batch_normalization_1_2926131batch_normalization_1_2926133*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2925802╙
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_transpose_1_2926136conv2d_transpose_1_2926138*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2925861°
leaky_re_lu_4/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2926079С
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_2_2926142batch_normalization_2_2926144batch_normalization_2_2926146batch_normalization_2_2926148*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2925906╙
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_transpose_2_2926151conv2d_transpose_2_2926153*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2925966К
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ¤
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:         d:         : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:'#
!
_user_specified_name	2926153:'#
!
_user_specified_name	2926151:'#
!
_user_specified_name	2926148:'#
!
_user_specified_name	2926146:'#
!
_user_specified_name	2926144:'#
!
_user_specified_name	2926142:'#
!
_user_specified_name	2926138:'#
!
_user_specified_name	2926136:'#
!
_user_specified_name	2926133:'#
!
_user_specified_name	2926131:'#
!
_user_specified_name	2926129:'#
!
_user_specified_name	2926127:'
#
!
_user_specified_name	2926123:'	#
!
_user_specified_name	2926121:'#
!
_user_specified_name	2926117:'#
!
_user_specified_name	2926115:'#
!
_user_specified_name	2926113:'#
!
_user_specified_name	2926111:'#
!
_user_specified_name	2926107:'#
!
_user_specified_name	2926105:'#
!
_user_specified_name	2926100:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:P L
'
_output_shapes
:         d
!
_user_specified_name	input_3
╤
Э
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2926744

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
р
b
F__inference_reshape_1_layer_call_and_return_conditional_losses_2926048

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :Ай
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:         Аa
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А1:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
┬
s
G__inference_multiply_1_layer_call_and_return_conditional_losses_2926388
inputs_0
inputs_1
identityP
mulMulinputs_0inputs_1*
T0*'
_output_shapes
:         dO
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         d:         d:QM
'
_output_shapes
:         d
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:         d
"
_user_specified_name
inputs_0
╔
K
/__inference_leaky_re_lu_4_layer_call_fn_2926677

inputs
identity╜
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2926079h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Щ

╓
7__inference_batch_normalization_1_layer_call_fn_2926581

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2925784К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926577:'#
!
_user_specified_name	2926575:'#
!
_user_specified_name	2926573:'#
!
_user_specified_name	2926571:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
■	
ў
D__inference_dense_1_layer_call_and_return_conditional_losses_2926014

inputs1
matmul_readvariableop_resource:	dА1.
biasadd_readvariableop_resource:	А1
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dА1*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А1s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А1*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А1`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         А1S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
°!
Ь
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2925966

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
▓&
э
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2925678

inputs6
'assignmovingavg_readvariableop_resource:	А18
)assignmovingavg_1_readvariableop_resource:	А14
%batchnorm_mul_readvariableop_resource:	А10
!batchnorm_readvariableop_resource:	А1
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А1*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А1И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А1*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А1*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А1*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А1*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А1y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А1м
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А1*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А1
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А1┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А1Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А1
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А1*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А1d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А1i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А1w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А1*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А1s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А1c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         А1╞
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А1: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
°!
Ь
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2926787

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
∙
ж
H__inference_embedding_2_layer_call_and_return_conditional_losses_2926365

inputs*
embedding_lookup_2926360:
d
identityИвembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:         ╜
embedding_lookupResourceGatherembedding_lookup_2926360Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2926360*+
_output_shapes
:         d*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:         du
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:         d5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:'#
!
_user_specified_name	2926360:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ш
f
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2926417

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         А1*
alpha%
╫#<`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         А1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А1:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
С

╥
7__inference_batch_normalization_2_layer_call_fn_2926695

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2925888Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926691:'#
!
_user_specified_name	2926689:'#
!
_user_specified_name	2926687:'#
!
_user_specified_name	2926685:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╛
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_2925996

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    d   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         dX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
·
к
2__inference_conv2d_transpose_layer_call_fn_2926525

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2925757К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926521:'#
!
_user_specified_name	2926519:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
РL
є

D__inference_model_1_layer_call_and_return_conditional_losses_2926096
input_3
input_4%
embedding_2_2925988:
d"
dense_1_2926015:	dА1
dense_1_2926017:	А1*
batch_normalization_2926026:	А1*
batch_normalization_2926028:	А1*
batch_normalization_2926030:	А1*
batch_normalization_2926032:	А14
conv2d_transpose_2926050:АА'
conv2d_transpose_2926052:	А,
batch_normalization_1_2926061:	А,
batch_normalization_1_2926063:	А,
batch_normalization_1_2926065:	А,
batch_normalization_1_2926067:	А5
conv2d_transpose_1_2926070:@А(
conv2d_transpose_1_2926072:@+
batch_normalization_2_2926081:@+
batch_normalization_2_2926083:@+
batch_normalization_2_2926085:@+
batch_normalization_2_2926087:@4
conv2d_transpose_2_2926090:@(
conv2d_transpose_2_2926092:
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв(conv2d_transpose/StatefulPartitionedCallв*conv2d_transpose_1/StatefulPartitionedCallв*conv2d_transpose_2/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв#embedding_2/StatefulPartitionedCallэ
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_4embedding_2_2925988*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_embedding_2_layer_call_and_return_conditional_losses_2925987с
flatten_3/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_2925996у
multiply_1/PartitionedCallPartitionedCallinput_3"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_multiply_1_layer_call_and_return_conditional_losses_2926003Н
dense_1/StatefulPartitionedCallStatefulPartitionedCall#multiply_1/PartitionedCall:output:0dense_1_2926015dense_1_2926017*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2926014ц
leaky_re_lu_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2926024№
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2926026batch_normalization_2926028batch_normalization_2926030batch_normalization_2926032*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2925678Є
reshape_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_2926048╕
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_2926050conv2d_transpose_2926052*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2925757ў
leaky_re_lu_3/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2926059Р
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_1_2926061batch_normalization_1_2926063batch_normalization_1_2926065batch_normalization_1_2926067*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2925784╙
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_transpose_1_2926070conv2d_transpose_1_2926072*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2925861°
leaky_re_lu_4/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2926079П
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_2_2926081batch_normalization_2_2926083batch_normalization_2_2926085batch_normalization_2_2926087*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2925888╙
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_transpose_2_2926090conv2d_transpose_2_2926092*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2925966К
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ¤
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:         d:         : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:'#
!
_user_specified_name	2926092:'#
!
_user_specified_name	2926090:'#
!
_user_specified_name	2926087:'#
!
_user_specified_name	2926085:'#
!
_user_specified_name	2926083:'#
!
_user_specified_name	2926081:'#
!
_user_specified_name	2926072:'#
!
_user_specified_name	2926070:'#
!
_user_specified_name	2926067:'#
!
_user_specified_name	2926065:'#
!
_user_specified_name	2926063:'#
!
_user_specified_name	2926061:'
#
!
_user_specified_name	2926052:'	#
!
_user_specified_name	2926050:'#
!
_user_specified_name	2926032:'#
!
_user_specified_name	2926030:'#
!
_user_specified_name	2926028:'#
!
_user_specified_name	2926026:'#
!
_user_specified_name	2926017:'#
!
_user_specified_name	2926015:'#
!
_user_specified_name	2925988:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:P L
'
_output_shapes
:         d
!
_user_specified_name	input_3
·
к
4__inference_conv2d_transpose_1_layer_call_fn_2926639

inputs"
unknown:@А
	unknown_0:@
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2925861Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926635:'#
!
_user_specified_name	2926633:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
о	
╘
5__inference_batch_normalization_layer_call_fn_2926443

inputs
unknown:	А1
	unknown_0:	А1
	unknown_1:	А1
	unknown_2:	А1
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2925698p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А1<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А1: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926439:'#
!
_user_specified_name	2926437:'#
!
_user_specified_name	2926435:'#
!
_user_specified_name	2926433:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
Д┼
┤
"__inference__wrapped_model_2925644
input_3
input_4>
,model_1_embedding_2_embedding_lookup_2925515:
dA
.model_1_dense_1_matmul_readvariableop_resource:	dА1>
/model_1_dense_1_biasadd_readvariableop_resource:	А1L
=model_1_batch_normalization_batchnorm_readvariableop_resource:	А1P
Amodel_1_batch_normalization_batchnorm_mul_readvariableop_resource:	А1N
?model_1_batch_normalization_batchnorm_readvariableop_1_resource:	А1N
?model_1_batch_normalization_batchnorm_readvariableop_2_resource:	А1]
Amodel_1_conv2d_transpose_conv2d_transpose_readvariableop_resource:ААG
8model_1_conv2d_transpose_biasadd_readvariableop_resource:	АD
5model_1_batch_normalization_1_readvariableop_resource:	АF
7model_1_batch_normalization_1_readvariableop_1_resource:	АU
Fmodel_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	АW
Hmodel_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	А^
Cmodel_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@АH
:model_1_conv2d_transpose_1_biasadd_readvariableop_resource:@C
5model_1_batch_normalization_2_readvariableop_resource:@E
7model_1_batch_normalization_2_readvariableop_1_resource:@T
Fmodel_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@V
Hmodel_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@]
Cmodel_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:@H
:model_1_conv2d_transpose_2_biasadd_readvariableop_resource:
identityИв4model_1/batch_normalization/batchnorm/ReadVariableOpв6model_1/batch_normalization/batchnorm/ReadVariableOp_1в6model_1/batch_normalization/batchnorm/ReadVariableOp_2в8model_1/batch_normalization/batchnorm/mul/ReadVariableOpв=model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpв?model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в,model_1/batch_normalization_1/ReadVariableOpв.model_1/batch_normalization_1/ReadVariableOp_1в=model_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpв?model_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в,model_1/batch_normalization_2/ReadVariableOpв.model_1/batch_normalization_2/ReadVariableOp_1в/model_1/conv2d_transpose/BiasAdd/ReadVariableOpв8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOpв1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOpв:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpв1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOpв:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpв&model_1/dense_1/BiasAdd/ReadVariableOpв%model_1/dense_1/MatMul/ReadVariableOpв$model_1/embedding_2/embedding_lookupj
model_1/embedding_2/CastCastinput_4*

DstT0*

SrcT0*'
_output_shapes
:         Н
$model_1/embedding_2/embedding_lookupResourceGather,model_1_embedding_2_embedding_lookup_2925515model_1/embedding_2/Cast:y:0*
Tindices0*?
_class5
31loc:@model_1/embedding_2/embedding_lookup/2925515*+
_output_shapes
:         d*
dtype0Ю
-model_1/embedding_2/embedding_lookup/IdentityIdentity-model_1/embedding_2/embedding_lookup:output:0*
T0*+
_output_shapes
:         dh
model_1/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"    d   ░
model_1/flatten_3/ReshapeReshape6model_1/embedding_2/embedding_lookup/Identity:output:0 model_1/flatten_3/Const:output:0*
T0*'
_output_shapes
:         d|
model_1/multiply_1/mulMulinput_3"model_1/flatten_3/Reshape:output:0*
T0*'
_output_shapes
:         dХ
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	dА1*
dtype0Ю
model_1/dense_1/MatMulMatMulmodel_1/multiply_1/mul:z:0-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А1У
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А1*
dtype0з
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А1И
model_1/leaky_re_lu_2/LeakyRelu	LeakyRelu model_1/dense_1/BiasAdd:output:0*(
_output_shapes
:         А1*
alpha%
╫#<п
4model_1/batch_normalization/batchnorm/ReadVariableOpReadVariableOp=model_1_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:А1*
dtype0p
+model_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╠
)model_1/batch_normalization/batchnorm/addAddV2<model_1/batch_normalization/batchnorm/ReadVariableOp:value:04model_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А1Й
+model_1/batch_normalization/batchnorm/RsqrtRsqrt-model_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:А1╖
8model_1/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_1_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А1*
dtype0╔
)model_1/batch_normalization/batchnorm/mulMul/model_1/batch_normalization/batchnorm/Rsqrt:y:0@model_1/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А1├
+model_1/batch_normalization/batchnorm/mul_1Mul-model_1/leaky_re_lu_2/LeakyRelu:activations:0-model_1/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А1│
6model_1/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp?model_1_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:А1*
dtype0╟
+model_1/batch_normalization/batchnorm/mul_2Mul>model_1/batch_normalization/batchnorm/ReadVariableOp_1:value:0-model_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:А1│
6model_1/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp?model_1_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:А1*
dtype0╟
)model_1/batch_normalization/batchnorm/subSub>model_1/batch_normalization/batchnorm/ReadVariableOp_2:value:0/model_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А1╟
+model_1/batch_normalization/batchnorm/add_1AddV2/model_1/batch_normalization/batchnorm/mul_1:z:0-model_1/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А1Д
model_1/reshape_1/ShapeShape/model_1/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::э╧o
%model_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
model_1/reshape_1/strided_sliceStridedSlice model_1/reshape_1/Shape:output:0.model_1/reshape_1/strided_slice/stack:output:00model_1/reshape_1/strided_slice/stack_1:output:00model_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!model_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d
!model_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :АГ
model_1/reshape_1/Reshape/shapePack(model_1/reshape_1/strided_slice:output:0*model_1/reshape_1/Reshape/shape/1:output:0*model_1/reshape_1/Reshape/shape/2:output:0*model_1/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:║
model_1/reshape_1/ReshapeReshape/model_1/batch_normalization/batchnorm/add_1:z:0(model_1/reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:         А~
model_1/conv2d_transpose/ShapeShape"model_1/reshape_1/Reshape:output:0*
T0*
_output_shapes
::э╧v
,model_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
&model_1/conv2d_transpose/strided_sliceStridedSlice'model_1/conv2d_transpose/Shape:output:05model_1/conv2d_transpose/strided_slice/stack:output:07model_1/conv2d_transpose/strided_slice/stack_1:output:07model_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 model_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 model_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :c
 model_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :АЖ
model_1/conv2d_transpose/stackPack/model_1/conv2d_transpose/strided_slice:output:0)model_1/conv2d_transpose/stack/1:output:0)model_1/conv2d_transpose/stack/2:output:0)model_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╓
(model_1/conv2d_transpose/strided_slice_1StridedSlice'model_1/conv2d_transpose/stack:output:07model_1/conv2d_transpose/strided_slice_1/stack:output:09model_1/conv2d_transpose/strided_slice_1/stack_1:output:09model_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask─
8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype0▓
)model_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput'model_1/conv2d_transpose/stack:output:0@model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0"model_1/reshape_1/Reshape:output:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
е
/model_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8model_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╙
 model_1/conv2d_transpose/BiasAddBiasAdd2model_1/conv2d_transpose/conv2d_transpose:output:07model_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЩ
model_1/leaky_re_lu_3/LeakyRelu	LeakyRelu)model_1/conv2d_transpose/BiasAdd:output:0*0
_output_shapes
:         А*
alpha%
╫#<Я
,model_1/batch_normalization_1/ReadVariableOpReadVariableOp5model_1_batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0г
.model_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0┴
=model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0┼
?model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0°
.model_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3-model_1/leaky_re_lu_3/LeakyRelu:activations:04model_1/batch_normalization_1/ReadVariableOp:value:06model_1/batch_normalization_1/ReadVariableOp_1:value:0Emodel_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( Р
 model_1/conv2d_transpose_1/ShapeShape2model_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
::э╧x
.model_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╪
(model_1/conv2d_transpose_1/strided_sliceStridedSlice)model_1/conv2d_transpose_1/Shape:output:07model_1/conv2d_transpose_1/strided_slice/stack:output:09model_1/conv2d_transpose_1/strided_slice/stack_1:output:09model_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"model_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"model_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@Р
 model_1/conv2d_transpose_1/stackPack1model_1/conv2d_transpose_1/strided_slice:output:0+model_1/conv2d_transpose_1/stack/1:output:0+model_1/conv2d_transpose_1/stack/2:output:0+model_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*model_1/conv2d_transpose_1/strided_slice_1StridedSlice)model_1/conv2d_transpose_1/stack:output:09model_1/conv2d_transpose_1/strided_slice_1/stack:output:0;model_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╟
:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype0╟
+model_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_1/stack:output:0Bmodel_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:02model_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
и
1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╪
"model_1/conv2d_transpose_1/BiasAddBiasAdd4model_1/conv2d_transpose_1/conv2d_transpose:output:09model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @Ъ
model_1/leaky_re_lu_4/LeakyRelu	LeakyRelu+model_1/conv2d_transpose_1/BiasAdd:output:0*/
_output_shapes
:         @*
alpha%
╫#<Ю
,model_1/batch_normalization_2/ReadVariableOpReadVariableOp5model_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0в
.model_1/batch_normalization_2/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0└
=model_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0─
?model_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0є
.model_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3-model_1/leaky_re_lu_4/LeakyRelu:activations:04model_1/batch_normalization_2/ReadVariableOp:value:06model_1/batch_normalization_2/ReadVariableOp_1:value:0Emodel_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( Р
 model_1/conv2d_transpose_2/ShapeShape2model_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
::э╧x
.model_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╪
(model_1/conv2d_transpose_2/strided_sliceStridedSlice)model_1/conv2d_transpose_2/Shape:output:07model_1/conv2d_transpose_2/strided_slice/stack:output:09model_1/conv2d_transpose_2/strided_slice/stack_1:output:09model_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_1/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"model_1/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"model_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Р
 model_1/conv2d_transpose_2/stackPack1model_1/conv2d_transpose_2/strided_slice:output:0+model_1/conv2d_transpose_2/stack/1:output:0+model_1/conv2d_transpose_2/stack/2:output:0+model_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*model_1/conv2d_transpose_2/strided_slice_1StridedSlice)model_1/conv2d_transpose_2/stack:output:09model_1/conv2d_transpose_2/strided_slice_1/stack:output:0;model_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╞
:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0╟
+model_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_2/stack:output:0Bmodel_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:02model_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
и
1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╪
"model_1/conv2d_transpose_2/BiasAddBiasAdd4model_1/conv2d_transpose_2/conv2d_transpose:output:09model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         О
model_1/conv2d_transpose_2/TanhTanh+model_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:         z
IdentityIdentity#model_1/conv2d_transpose_2/Tanh:y:0^NoOp*
T0*/
_output_shapes
:         С	
NoOpNoOp5^model_1/batch_normalization/batchnorm/ReadVariableOp7^model_1/batch_normalization/batchnorm/ReadVariableOp_17^model_1/batch_normalization/batchnorm/ReadVariableOp_29^model_1/batch_normalization/batchnorm/mul/ReadVariableOp>^model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_1/ReadVariableOp/^model_1/batch_normalization_1/ReadVariableOp_1>^model_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_2/ReadVariableOp/^model_1/batch_normalization_2/ReadVariableOp_10^model_1/conv2d_transpose/BiasAdd/ReadVariableOp9^model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp'^model_1/dense_1/BiasAdd/ReadVariableOp&^model_1/dense_1/MatMul/ReadVariableOp%^model_1/embedding_2/embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:         d:         : : : : : : : : : : : : : : : : : : : : : 2p
6model_1/batch_normalization/batchnorm/ReadVariableOp_16model_1/batch_normalization/batchnorm/ReadVariableOp_12p
6model_1/batch_normalization/batchnorm/ReadVariableOp_26model_1/batch_normalization/batchnorm/ReadVariableOp_22l
4model_1/batch_normalization/batchnorm/ReadVariableOp4model_1/batch_normalization/batchnorm/ReadVariableOp2t
8model_1/batch_normalization/batchnorm/mul/ReadVariableOp8model_1/batch_normalization/batchnorm/mul/ReadVariableOp2В
?model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12~
=model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2`
.model_1/batch_normalization_1/ReadVariableOp_1.model_1/batch_normalization_1/ReadVariableOp_12\
,model_1/batch_normalization_1/ReadVariableOp,model_1/batch_normalization_1/ReadVariableOp2В
?model_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12~
=model_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2`
.model_1/batch_normalization_2/ReadVariableOp_1.model_1/batch_normalization_2/ReadVariableOp_12\
,model_1/batch_normalization_2/ReadVariableOp,model_1/batch_normalization_2/ReadVariableOp2b
/model_1/conv2d_transpose/BiasAdd/ReadVariableOp/model_1/conv2d_transpose/BiasAdd/ReadVariableOp2t
8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2P
&model_1/dense_1/BiasAdd/ReadVariableOp&model_1/dense_1/BiasAdd/ReadVariableOp2N
%model_1/dense_1/MatMul/ReadVariableOp%model_1/dense_1/MatMul/ReadVariableOp2L
$model_1/embedding_2/embedding_lookup$model_1/embedding_2/embedding_lookup:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:'#
!
_user_specified_name	2925515:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:P L
'
_output_shapes
:         d
!
_user_specified_name	input_3
╤
Э
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2925906

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
р
b
F__inference_reshape_1_layer_call_and_return_conditional_losses_2926516

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :Ай
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:         Аa
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А1:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
Ш!
Э
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2926672

inputsC
(conv2d_transpose_readvariableop_resource:@А-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ж
╗
%__inference_signature_wrapper_2926349
input_3
input_4
unknown:
d
	unknown_0:	dА1
	unknown_1:	А1
	unknown_2:	А1
	unknown_3:	А1
	unknown_4:	А1
	unknown_5:	А1%
	unknown_6:АА
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А%

unknown_12:@А

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@$

unknown_18:@

unknown_19:
identityИвStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_2925644w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:         d:         : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926345:'#
!
_user_specified_name	2926343:'#
!
_user_specified_name	2926341:'#
!
_user_specified_name	2926339:'#
!
_user_specified_name	2926337:'#
!
_user_specified_name	2926335:'#
!
_user_specified_name	2926333:'#
!
_user_specified_name	2926331:'#
!
_user_specified_name	2926329:'#
!
_user_specified_name	2926327:'#
!
_user_specified_name	2926325:'#
!
_user_specified_name	2926323:'
#
!
_user_specified_name	2926321:'	#
!
_user_specified_name	2926319:'#
!
_user_specified_name	2926317:'#
!
_user_specified_name	2926315:'#
!
_user_specified_name	2926313:'#
!
_user_specified_name	2926311:'#
!
_user_specified_name	2926309:'#
!
_user_specified_name	2926307:'#
!
_user_specified_name	2926305:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:P L
'
_output_shapes
:         d
!
_user_specified_name	input_3
Ёо
Ї
 __inference__traced_save_2926936
file_prefix?
-read_disablecopyonread_embedding_2_embeddings:
d:
'read_1_disablecopyonread_dense_1_kernel:	dА14
%read_2_disablecopyonread_dense_1_bias:	А1A
2read_3_disablecopyonread_batch_normalization_gamma:	А1@
1read_4_disablecopyonread_batch_normalization_beta:	А1G
8read_5_disablecopyonread_batch_normalization_moving_mean:	А1K
<read_6_disablecopyonread_batch_normalization_moving_variance:	А1L
0read_7_disablecopyonread_conv2d_transpose_kernel:АА=
.read_8_disablecopyonread_conv2d_transpose_bias:	АC
4read_9_disablecopyonread_batch_normalization_1_gamma:	АC
4read_10_disablecopyonread_batch_normalization_1_beta:	АJ
;read_11_disablecopyonread_batch_normalization_1_moving_mean:	АN
?read_12_disablecopyonread_batch_normalization_1_moving_variance:	АN
3read_13_disablecopyonread_conv2d_transpose_1_kernel:@А?
1read_14_disablecopyonread_conv2d_transpose_1_bias:@C
5read_15_disablecopyonread_batch_normalization_2_gamma:@B
4read_16_disablecopyonread_batch_normalization_2_beta:@I
;read_17_disablecopyonread_batch_normalization_2_moving_mean:@M
?read_18_disablecopyonread_batch_normalization_2_moving_variance:@M
3read_19_disablecopyonread_conv2d_transpose_2_kernel:@?
1read_20_disablecopyonread_conv2d_transpose_2_bias:
savev2_const
identity_43ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
Read/DisableCopyOnReadDisableCopyOnRead-read_disablecopyonread_embedding_2_embeddings"/device:CPU:0*
_output_shapes
 й
Read/ReadVariableOpReadVariableOp-read_disablecopyonread_embedding_2_embeddings^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
d*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
da

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:
d{
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 и
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_dense_1_kernel^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	dА1*
dtype0n

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	dА1d

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:	dА1y
Read_2/DisableCopyOnReadDisableCopyOnRead%read_2_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 в
Read_2/ReadVariableOpReadVariableOp%read_2_disablecopyonread_dense_1_bias^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А1*
dtype0j

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А1`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:А1Ж
Read_3/DisableCopyOnReadDisableCopyOnRead2read_3_disablecopyonread_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 п
Read_3/ReadVariableOpReadVariableOp2read_3_disablecopyonread_batch_normalization_gamma^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А1*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А1`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:А1Е
Read_4/DisableCopyOnReadDisableCopyOnRead1read_4_disablecopyonread_batch_normalization_beta"/device:CPU:0*
_output_shapes
 о
Read_4/ReadVariableOpReadVariableOp1read_4_disablecopyonread_batch_normalization_beta^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А1*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А1`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:А1М
Read_5/DisableCopyOnReadDisableCopyOnRead8read_5_disablecopyonread_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 ╡
Read_5/ReadVariableOpReadVariableOp8read_5_disablecopyonread_batch_normalization_moving_mean^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А1*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А1b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:А1Р
Read_6/DisableCopyOnReadDisableCopyOnRead<read_6_disablecopyonread_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 ╣
Read_6/ReadVariableOpReadVariableOp<read_6_disablecopyonread_batch_normalization_moving_variance^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А1*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А1b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:А1Д
Read_7/DisableCopyOnReadDisableCopyOnRead0read_7_disablecopyonread_conv2d_transpose_kernel"/device:CPU:0*
_output_shapes
 ║
Read_7/ReadVariableOpReadVariableOp0read_7_disablecopyonread_conv2d_transpose_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:АА*
dtype0x
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ААo
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*(
_output_shapes
:ААВ
Read_8/DisableCopyOnReadDisableCopyOnRead.read_8_disablecopyonread_conv2d_transpose_bias"/device:CPU:0*
_output_shapes
 л
Read_8/ReadVariableOpReadVariableOp.read_8_disablecopyonread_conv2d_transpose_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:АИ
Read_9/DisableCopyOnReadDisableCopyOnRead4read_9_disablecopyonread_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 ▒
Read_9/ReadVariableOpReadVariableOp4read_9_disablecopyonread_batch_normalization_1_gamma^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЙ
Read_10/DisableCopyOnReadDisableCopyOnRead4read_10_disablecopyonread_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 │
Read_10/ReadVariableOpReadVariableOp4read_10_disablecopyonread_batch_normalization_1_beta^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:АР
Read_11/DisableCopyOnReadDisableCopyOnRead;read_11_disablecopyonread_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 ║
Read_11/ReadVariableOpReadVariableOp;read_11_disablecopyonread_batch_normalization_1_moving_mean^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:АФ
Read_12/DisableCopyOnReadDisableCopyOnRead?read_12_disablecopyonread_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 ╛
Read_12/ReadVariableOpReadVariableOp?read_12_disablecopyonread_batch_normalization_1_moving_variance^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:АИ
Read_13/DisableCopyOnReadDisableCopyOnRead3read_13_disablecopyonread_conv2d_transpose_1_kernel"/device:CPU:0*
_output_shapes
 ╛
Read_13/ReadVariableOpReadVariableOp3read_13_disablecopyonread_conv2d_transpose_1_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@А*
dtype0x
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@Аn
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*'
_output_shapes
:@АЖ
Read_14/DisableCopyOnReadDisableCopyOnRead1read_14_disablecopyonread_conv2d_transpose_1_bias"/device:CPU:0*
_output_shapes
 п
Read_14/ReadVariableOpReadVariableOp1read_14_disablecopyonread_conv2d_transpose_1_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@К
Read_15/DisableCopyOnReadDisableCopyOnRead5read_15_disablecopyonread_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 │
Read_15/ReadVariableOpReadVariableOp5read_15_disablecopyonread_batch_normalization_2_gamma^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@Й
Read_16/DisableCopyOnReadDisableCopyOnRead4read_16_disablecopyonread_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 ▓
Read_16/ReadVariableOpReadVariableOp4read_16_disablecopyonread_batch_normalization_2_beta^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_17/DisableCopyOnReadDisableCopyOnRead;read_17_disablecopyonread_batch_normalization_2_moving_mean"/device:CPU:0*
_output_shapes
 ╣
Read_17/ReadVariableOpReadVariableOp;read_17_disablecopyonread_batch_normalization_2_moving_mean^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_18/DisableCopyOnReadDisableCopyOnRead?read_18_disablecopyonread_batch_normalization_2_moving_variance"/device:CPU:0*
_output_shapes
 ╜
Read_18/ReadVariableOpReadVariableOp?read_18_disablecopyonread_batch_normalization_2_moving_variance^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:@И
Read_19/DisableCopyOnReadDisableCopyOnRead3read_19_disablecopyonread_conv2d_transpose_2_kernel"/device:CPU:0*
_output_shapes
 ╜
Read_19/ReadVariableOpReadVariableOp3read_19_disablecopyonread_conv2d_transpose_2_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*&
_output_shapes
:@Ж
Read_20/DisableCopyOnReadDisableCopyOnRead1read_20_disablecopyonread_conv2d_transpose_2_bias"/device:CPU:0*
_output_shapes
 п
Read_20/ReadVariableOpReadVariableOp1read_20_disablecopyonread_conv2d_transpose_2_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:┐

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ш	
value▐	B█	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЩ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B ╢
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *$
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_42Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_43IdentityIdentity_42:output:0^NoOp*
T0*
_output_shapes
: °
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_43Identity_43:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:73
1
_user_specified_nameconv2d_transpose_2/bias:95
3
_user_specified_nameconv2d_transpose_2/kernel:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:A=
;
_user_specified_name#!batch_normalization_2/moving_mean::6
4
_user_specified_namebatch_normalization_2/beta:;7
5
_user_specified_namebatch_normalization_2/gamma:73
1
_user_specified_nameconv2d_transpose_1/bias:95
3
_user_specified_nameconv2d_transpose_1/kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::6
4
_user_specified_namebatch_normalization_1/beta:;
7
5
_user_specified_namebatch_normalization_1/gamma:5	1
/
_user_specified_nameconv2d_transpose/bias:73
1
_user_specified_nameconv2d_transpose/kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:62
0
_user_specified_nameembedding_2/embeddings:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
с
б
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2925802

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           АМ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ш!
Э
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2925861

inputsC
(conv2d_transpose_readvariableop_resource:@А-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ї
Ш
)__inference_dense_1_layer_call_fn_2926397

inputs
unknown:	dА1
	unknown_0:	А1
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2926014p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А1<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926393:'#
!
_user_specified_name	2926391:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╠
┐
)__inference_model_1_layer_call_fn_2926253
input_3
input_4
unknown:
d
	unknown_0:	dА1
	unknown_1:	А1
	unknown_2:	А1
	unknown_3:	А1
	unknown_4:	А1
	unknown_5:	А1%
	unknown_6:АА
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А%

unknown_12:@А

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@$

unknown_18:@

unknown_19:
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2926157w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:         d:         : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926249:'#
!
_user_specified_name	2926247:'#
!
_user_specified_name	2926245:'#
!
_user_specified_name	2926243:'#
!
_user_specified_name	2926241:'#
!
_user_specified_name	2926239:'#
!
_user_specified_name	2926237:'#
!
_user_specified_name	2926235:'#
!
_user_specified_name	2926233:'#
!
_user_specified_name	2926231:'#
!
_user_specified_name	2926229:'#
!
_user_specified_name	2926227:'
#
!
_user_specified_name	2926225:'	#
!
_user_specified_name	2926223:'#
!
_user_specified_name	2926221:'#
!
_user_specified_name	2926219:'#
!
_user_specified_name	2926217:'#
!
_user_specified_name	2926215:'#
!
_user_specified_name	2926213:'#
!
_user_specified_name	2926211:'#
!
_user_specified_name	2926209:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:P L
'
_output_shapes
:         d
!
_user_specified_name	input_3
н
K
/__inference_leaky_re_lu_2_layer_call_fn_2926412

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2926024a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А1:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
Юi
╙
#__inference__traced_restore_2927008
file_prefix9
'assignvariableop_embedding_2_embeddings:
d4
!assignvariableop_1_dense_1_kernel:	dА1.
assignvariableop_2_dense_1_bias:	А1;
,assignvariableop_3_batch_normalization_gamma:	А1:
+assignvariableop_4_batch_normalization_beta:	А1A
2assignvariableop_5_batch_normalization_moving_mean:	А1E
6assignvariableop_6_batch_normalization_moving_variance:	А1F
*assignvariableop_7_conv2d_transpose_kernel:АА7
(assignvariableop_8_conv2d_transpose_bias:	А=
.assignvariableop_9_batch_normalization_1_gamma:	А=
.assignvariableop_10_batch_normalization_1_beta:	АD
5assignvariableop_11_batch_normalization_1_moving_mean:	АH
9assignvariableop_12_batch_normalization_1_moving_variance:	АH
-assignvariableop_13_conv2d_transpose_1_kernel:@А9
+assignvariableop_14_conv2d_transpose_1_bias:@=
/assignvariableop_15_batch_normalization_2_gamma:@<
.assignvariableop_16_batch_normalization_2_beta:@C
5assignvariableop_17_batch_normalization_2_moving_mean:@G
9assignvariableop_18_batch_normalization_2_moving_variance:@G
-assignvariableop_19_conv2d_transpose_2_kernel:@9
+assignvariableop_20_conv2d_transpose_2_bias:
identity_22ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9┬

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ш	
value▐	B█	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЬ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B М
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOpAssignVariableOp'assignvariableop_embedding_2_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_1_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_1_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_3AssignVariableOp,assignvariableop_3_batch_normalization_gammaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_4AssignVariableOp+assignvariableop_4_batch_normalization_betaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╔
AssignVariableOp_5AssignVariableOp2assignvariableop_5_batch_normalization_moving_meanIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_6AssignVariableOp6assignvariableop_6_batch_normalization_moving_varianceIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_7AssignVariableOp*assignvariableop_7_conv2d_transpose_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_8AssignVariableOp(assignvariableop_8_conv2d_transpose_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_1_gammaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_10AssignVariableOp.assignvariableop_10_batch_normalization_1_betaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_11AssignVariableOp5assignvariableop_11_batch_normalization_1_moving_meanIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_12AssignVariableOp9assignvariableop_12_batch_normalization_1_moving_varianceIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_13AssignVariableOp-assignvariableop_13_conv2d_transpose_1_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_14AssignVariableOp+assignvariableop_14_conv2d_transpose_1_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_2_gammaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_16AssignVariableOp.assignvariableop_16_batch_normalization_2_betaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_17AssignVariableOp5assignvariableop_17_batch_normalization_2_moving_meanIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_18AssignVariableOp9assignvariableop_18_batch_normalization_2_moving_varianceIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_19AssignVariableOp-assignvariableop_19_conv2d_transpose_2_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_20AssignVariableOp+assignvariableop_20_conv2d_transpose_2_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Э
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: ц
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_22Identity_22:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:73
1
_user_specified_nameconv2d_transpose_2/bias:95
3
_user_specified_nameconv2d_transpose_2/kernel:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:A=
;
_user_specified_name#!batch_normalization_2/moving_mean::6
4
_user_specified_namebatch_normalization_2/beta:;7
5
_user_specified_namebatch_normalization_2/gamma:73
1
_user_specified_nameconv2d_transpose_1/bias:95
3
_user_specified_nameconv2d_transpose_1/kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::6
4
_user_specified_namebatch_normalization_1/beta:;
7
5
_user_specified_namebatch_normalization_1/gamma:5	1
/
_user_specified_nameconv2d_transpose/bias:73
1
_user_specified_nameconv2d_transpose/kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:62
0
_user_specified_nameembedding_2/embeddings:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
║
q
G__inference_multiply_1_layer_call_and_return_conditional_losses_2926003

inputs
inputs_1
identityN
mulMulinputsinputs_1*
T0*'
_output_shapes
:         dO
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         d:         d:OK
'
_output_shapes
:         d
 
_user_specified_nameinputs:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
й
G
+__inference_flatten_3_layer_call_fn_2926370

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_2925996`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
Д
f
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2926079

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @*
alpha%
╫#<g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
у
│
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2925698

inputs0
!batchnorm_readvariableop_resource:	А14
%batchnorm_mul_readvariableop_resource:	А12
#batchnorm_readvariableop_1_resource:	А12
#batchnorm_readvariableop_2_resource:	А1
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А1*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А1Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А1
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А1*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А1d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А1{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А1*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А1{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А1*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А1s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А1c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         А1Ц
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А1: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
и
X
,__inference_multiply_1_layer_call_fn_2926382
inputs_0
inputs_1
identity┐
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_multiply_1_layer_call_and_return_conditional_losses_2926003`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         d:         d:QM
'
_output_shapes
:         d
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:         d
"
_user_specified_name
inputs_0
Ы

╓
7__inference_batch_normalization_1_layer_call_fn_2926594

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2925802К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926590:'#
!
_user_specified_name	2926588:'#
!
_user_specified_name	2926586:'#
!
_user_specified_name	2926584:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
п
Б
-__inference_embedding_2_layer_call_fn_2926356

inputs
unknown:
d
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_embedding_2_layer_call_and_return_conditional_losses_2925987s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926352:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ю!
Э
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2926558

inputsD
(conv2d_transpose_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :Аy
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype0▌
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ъ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╞
┐
)__inference_model_1_layer_call_fn_2926205
input_3
input_4
unknown:
d
	unknown_0:	dА1
	unknown_1:	А1
	unknown_2:	А1
	unknown_3:	А1
	unknown_4:	А1
	unknown_5:	А1%
	unknown_6:АА
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А%

unknown_12:@А

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@$

unknown_18:@

unknown_19:
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2926096w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:         d:         : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926201:'#
!
_user_specified_name	2926199:'#
!
_user_specified_name	2926197:'#
!
_user_specified_name	2926195:'#
!
_user_specified_name	2926193:'#
!
_user_specified_name	2926191:'#
!
_user_specified_name	2926189:'#
!
_user_specified_name	2926187:'#
!
_user_specified_name	2926185:'#
!
_user_specified_name	2926183:'#
!
_user_specified_name	2926181:'#
!
_user_specified_name	2926179:'
#
!
_user_specified_name	2926177:'	#
!
_user_specified_name	2926175:'#
!
_user_specified_name	2926173:'#
!
_user_specified_name	2926171:'#
!
_user_specified_name	2926169:'#
!
_user_specified_name	2926167:'#
!
_user_specified_name	2926165:'#
!
_user_specified_name	2926163:'#
!
_user_specified_name	2926161:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:P L
'
_output_shapes
:         d
!
_user_specified_name	input_3
с
б
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2926630

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           АМ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Л
┴
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2926726

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%═╠L>╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ю!
Э
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2925757

inputsD
(conv2d_transpose_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :Аy
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype0▌
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ъ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
И
f
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2926568

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:         А*
alpha%
╫#<h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Д
f
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2926682

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @*
alpha%
╫#<g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
▓&
э
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2926477

inputs6
'assignmovingavg_readvariableop_resource:	А18
)assignmovingavg_1_readvariableop_resource:	А14
%batchnorm_mul_readvariableop_resource:	А10
!batchnorm_readvariableop_resource:	А1
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А1*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А1И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А1*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А1*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А1*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А1*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А1y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А1м
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А1*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А1
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А1┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А1Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А1
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А1*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А1d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А1i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А1w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А1*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А1s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А1c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         А1╞
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А1: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
Л
┴
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2925888

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%═╠L>╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
И
f
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2926059

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:         А*
alpha%
╫#<h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ў
й
4__inference_conv2d_transpose_2_layer_call_fn_2926753

inputs!
unknown:@
	unknown_0:
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2925966Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926749:'#
!
_user_specified_name	2926747:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╛
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_2926376

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    d   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         dX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
∙
ж
H__inference_embedding_2_layer_call_and_return_conditional_losses_2925987

inputs*
embedding_lookup_2925982:
d
identityИвembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:         ╜
embedding_lookupResourceGatherembedding_lookup_2925982Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2925982*+
_output_shapes
:         d*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:         du
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:         d5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:'#
!
_user_specified_name	2925982:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╡
G
+__inference_reshape_1_layer_call_fn_2926502

inputs
identity║
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_2926048i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А1:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
м	
╘
5__inference_batch_normalization_layer_call_fn_2926430

inputs
unknown:	А1
	unknown_0:	А1
	unknown_1:	А1
	unknown_2:	А1
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2925678p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А1<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А1: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2926426:'#
!
_user_specified_name	2926424:'#
!
_user_specified_name	2926422:'#
!
_user_specified_name	2926420:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs
■	
ў
D__inference_dense_1_layer_call_and_return_conditional_losses_2926407

inputs1
matmul_readvariableop_resource:	dА1.
biasadd_readvariableop_resource:	А1
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dА1*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А1s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А1*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А1`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         А1S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
═
K
/__inference_leaky_re_lu_3_layer_call_fn_2926563

inputs
identity╛
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2926059i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
у
│
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2926497

inputs0
!batchnorm_readvariableop_resource:	А14
%batchnorm_mul_readvariableop_resource:	А12
#batchnorm_readvariableop_1_resource:	А12
#batchnorm_readvariableop_2_resource:	А1
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А1*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А1Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А1
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А1*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А1d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А1{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А1*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А1{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А1*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А1s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А1c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         А1Ц
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А1: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А1
 
_user_specified_nameinputs"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*·
serving_defaultц
;
input_30
serving_default_input_3:0         d
;
input_40
serving_default_input_4:0         N
conv2d_transpose_28
StatefulPartitionedCall:0         tensorflow/serving/predict:╦┐
·
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
╡
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
"
_tf_keras_input_layer
е
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
е
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias"
_tf_keras_layer
е
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
ъ
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance"
_tf_keras_layer
е
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias
 S_jit_compiled_convolution_op"
_tf_keras_layer
е
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
ъ
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
`axis
	agamma
bbeta
cmoving_mean
dmoving_variance"
_tf_keras_layer
▌
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias
 m_jit_compiled_convolution_op"
_tf_keras_layer
е
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
ъ
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses
zaxis
	{gamma
|beta
}moving_mean
~moving_variance"
_tf_keras_layer
х
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses
Еkernel
	Жbias
!З_jit_compiled_convolution_op"
_tf_keras_layer
└
0
21
32
A3
B4
C5
D6
Q7
R8
a9
b10
c11
d12
k13
l14
{15
|16
}17
~18
Е19
Ж20"
trackable_list_wrapper
Р
0
21
32
A3
B4
Q5
R6
a7
b8
k9
l10
{11
|12
Е13
Ж14"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╔
Нtrace_0
Оtrace_12О
)__inference_model_1_layer_call_fn_2926205
)__inference_model_1_layer_call_fn_2926253╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0zОtrace_1
 
Пtrace_0
Рtrace_12─
D__inference_model_1_layer_call_and_return_conditional_losses_2926096
D__inference_model_1_layer_call_and_return_conditional_losses_2926157╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zПtrace_0zРtrace_1
╓B╙
"__inference__wrapped_model_2925644input_3input_4"Ш
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
-
Сserving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
щ
Чtrace_02╩
-__inference_embedding_2_layer_call_fn_2926356Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЧtrace_0
Д
Шtrace_02х
H__inference_embedding_2_layer_call_and_return_conditional_losses_2926365Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zШtrace_0
(:&
d2embedding_2/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ч
Юtrace_02╚
+__inference_flatten_3_layer_call_fn_2926370Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЮtrace_0
В
Яtrace_02у
F__inference_flatten_3_layer_call_and_return_conditional_losses_2926376Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЯtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
аnon_trainable_variables
бlayers
вmetrics
 гlayer_regularization_losses
дlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
ш
еtrace_02╔
,__inference_multiply_1_layer_call_fn_2926382Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zеtrace_0
Г
жtrace_02ф
G__inference_multiply_1_layer_call_and_return_conditional_losses_2926388Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zжtrace_0
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
х
мtrace_02╞
)__inference_dense_1_layer_call_fn_2926397Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zмtrace_0
А
нtrace_02с
D__inference_dense_1_layer_call_and_return_conditional_losses_2926407Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zнtrace_0
!:	dА12dense_1/kernel
:А12dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
оnon_trainable_variables
пlayers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
ы
│trace_02╠
/__inference_leaky_re_lu_2_layer_call_fn_2926412Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z│trace_0
Ж
┤trace_02ч
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2926417Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┤trace_0
<
A0
B1
C2
D3"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
с
║trace_0
╗trace_12ж
5__inference_batch_normalization_layer_call_fn_2926430
5__inference_batch_normalization_layer_call_fn_2926443╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z║trace_0z╗trace_1
Ч
╝trace_0
╜trace_12▄
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2926477
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2926497╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╝trace_0z╜trace_1
 "
trackable_list_wrapper
(:&А12batch_normalization/gamma
':%А12batch_normalization/beta
0:.А1 (2batch_normalization/moving_mean
4:2А1 (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
ч
├trace_02╚
+__inference_reshape_1_layer_call_fn_2926502Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z├trace_0
В
─trace_02у
F__inference_reshape_1_layer_call_and_return_conditional_losses_2926516Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z─trace_0
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
ю
╩trace_02╧
2__inference_conv2d_transpose_layer_call_fn_2926525Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0
Й
╦trace_02ъ
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2926558Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╦trace_0
3:1АА2conv2d_transpose/kernel
$:"А2conv2d_transpose/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
ы
╤trace_02╠
/__inference_leaky_re_lu_3_layer_call_fn_2926563Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╤trace_0
Ж
╥trace_02ч
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2926568Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0
<
a0
b1
c2
d3"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
х
╪trace_0
┘trace_12к
7__inference_batch_normalization_1_layer_call_fn_2926581
7__inference_batch_normalization_1_layer_call_fn_2926594╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╪trace_0z┘trace_1
Ы
┌trace_0
█trace_12р
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2926612
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2926630╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┌trace_0z█trace_1
 "
trackable_list_wrapper
*:(А2batch_normalization_1/gamma
):'А2batch_normalization_1/beta
2:0А (2!batch_normalization_1/moving_mean
6:4А (2%batch_normalization_1/moving_variance
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▄non_trainable_variables
▌layers
▐metrics
 ▀layer_regularization_losses
рlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
Ё
сtrace_02╤
4__inference_conv2d_transpose_1_layer_call_fn_2926639Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zсtrace_0
Л
тtrace_02ь
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2926672Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zтtrace_0
4:2@А2conv2d_transpose_1/kernel
%:#@2conv2d_transpose_1/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
ы
шtrace_02╠
/__inference_leaky_re_lu_4_layer_call_fn_2926677Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zшtrace_0
Ж
щtrace_02ч
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2926682Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zщtrace_0
<
{0
|1
}2
~3"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
х
яtrace_0
Ёtrace_12к
7__inference_batch_normalization_2_layer_call_fn_2926695
7__inference_batch_normalization_2_layer_call_fn_2926708╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zяtrace_0zЁtrace_1
Ы
ёtrace_0
Єtrace_12р
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2926726
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2926744╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zёtrace_0zЄtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
0
Е0
Ж1"
trackable_list_wrapper
0
Е0
Ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
╖
єnon_trainable_variables
Їlayers
їmetrics
 Ўlayer_regularization_losses
ўlayer_metrics
	variables
Аtrainable_variables
Бregularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
Ё
°trace_02╤
4__inference_conv2d_transpose_2_layer_call_fn_2926753Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z°trace_0
Л
∙trace_02ь
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2926787Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z∙trace_0
3:1@2conv2d_transpose_2/kernel
%:#2conv2d_transpose_2/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
J
C0
D1
c2
d3
}4
~5"
trackable_list_wrapper
Ц
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBю
)__inference_model_1_layer_call_fn_2926205input_3input_4"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
)__inference_model_1_layer_call_fn_2926253input_3input_4"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
D__inference_model_1_layer_call_and_return_conditional_losses_2926096input_3input_4"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
D__inference_model_1_layer_call_and_return_conditional_losses_2926157input_3input_4"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
уBр
%__inference_signature_wrapper_2926349input_3input_4"д
Э▓Щ
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 '

kwonlyargsЪ
	jinput_3
	jinput_4
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╫B╘
-__inference_embedding_2_layer_call_fn_2926356inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЄBя
H__inference_embedding_2_layer_call_and_return_conditional_losses_2926365inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╒B╥
+__inference_flatten_3_layer_call_fn_2926370inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
F__inference_flatten_3_layer_call_and_return_conditional_losses_2926376inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тB▀
,__inference_multiply_1_layer_call_fn_2926382inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
G__inference_multiply_1_layer_call_and_return_conditional_losses_2926388inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╙B╨
)__inference_dense_1_layer_call_fn_2926397inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
D__inference_dense_1_layer_call_and_return_conditional_losses_2926407inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
/__inference_leaky_re_lu_2_layer_call_fn_2926412inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2926417inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
єBЁ
5__inference_batch_normalization_layer_call_fn_2926430inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
єBЁ
5__inference_batch_normalization_layer_call_fn_2926443inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2926477inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2926497inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╒B╥
+__inference_reshape_1_layer_call_fn_2926502inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
F__inference_reshape_1_layer_call_and_return_conditional_losses_2926516inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▄B┘
2__inference_conv2d_transpose_layer_call_fn_2926525inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2926558inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
/__inference_leaky_re_lu_3_layer_call_fn_2926563inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2926568inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
їBЄ
7__inference_batch_normalization_1_layer_call_fn_2926581inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
7__inference_batch_normalization_1_layer_call_fn_2926594inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
РBН
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2926612inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
РBН
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2926630inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▐B█
4__inference_conv2d_transpose_1_layer_call_fn_2926639inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2926672inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
/__inference_leaky_re_lu_4_layer_call_fn_2926677inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2926682inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
їBЄ
7__inference_batch_normalization_2_layer_call_fn_2926695inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
7__inference_batch_normalization_2_layer_call_fn_2926708inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
РBН
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2926726inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
РBН
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2926744inputs"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▐B█
4__inference_conv2d_transpose_2_layer_call_fn_2926753inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2926787inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ы
"__inference__wrapped_model_2925644─23DACBQRabcdkl{|}~ЕЖXвU
NвK
IЪF
!К
input_3         d
!К
input_4         
к "OкL
J
conv2d_transpose_24К1
conv2d_transpose_2         ·
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2926612гabcdRвO
HвE
;К8
inputs,                           А
p

 
к "GвD
=К:
tensor_0,                           А
Ъ ·
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2926630гabcdRвO
HвE
;К8
inputs,                           А
p 

 
к "GвD
=К:
tensor_0,                           А
Ъ ╘
7__inference_batch_normalization_1_layer_call_fn_2926581ШabcdRвO
HвE
;К8
inputs,                           А
p

 
к "<К9
unknown,                           А╘
7__inference_batch_normalization_1_layer_call_fn_2926594ШabcdRвO
HвE
;К8
inputs,                           А
p 

 
к "<К9
unknown,                           А°
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2926726б{|}~QвN
GвD
:К7
inputs+                           @
p

 
к "FвC
<К9
tensor_0+                           @
Ъ °
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2926744б{|}~QвN
GвD
:К7
inputs+                           @
p 

 
к "FвC
<К9
tensor_0+                           @
Ъ ╥
7__inference_batch_normalization_2_layer_call_fn_2926695Ц{|}~QвN
GвD
:К7
inputs+                           @
p

 
к ";К8
unknown+                           @╥
7__inference_batch_normalization_2_layer_call_fn_2926708Ц{|}~QвN
GвD
:К7
inputs+                           @
p 

 
к ";К8
unknown+                           @├
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2926477oCDAB8в5
.в+
!К
inputs         А1
p

 
к "-в*
#К 
tensor_0         А1
Ъ ├
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2926497oDACB8в5
.в+
!К
inputs         А1
p 

 
к "-в*
#К 
tensor_0         А1
Ъ Э
5__inference_batch_normalization_layer_call_fn_2926430dCDAB8в5
.в+
!К
inputs         А1
p

 
к ""К
unknown         А1Э
5__inference_batch_normalization_layer_call_fn_2926443dDACB8в5
.в+
!К
inputs         А1
p 

 
к ""К
unknown         А1ь
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2926672ШklJвG
@в=
;К8
inputs,                           А
к "FвC
<К9
tensor_0+                           @
Ъ ╞
4__inference_conv2d_transpose_1_layer_call_fn_2926639НklJвG
@в=
;К8
inputs,                           А
к ";К8
unknown+                           @э
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2926787ЩЕЖIвF
?в<
:К7
inputs+                           @
к "FвC
<К9
tensor_0+                           
Ъ ╟
4__inference_conv2d_transpose_2_layer_call_fn_2926753ОЕЖIвF
?в<
:К7
inputs+                           @
к ";К8
unknown+                           ы
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2926558ЩQRJвG
@в=
;К8
inputs,                           А
к "GвD
=К:
tensor_0,                           А
Ъ ┼
2__inference_conv2d_transpose_layer_call_fn_2926525ОQRJвG
@в=
;К8
inputs,                           А
к "<К9
unknown,                           Ам
D__inference_dense_1_layer_call_and_return_conditional_losses_2926407d23/в,
%в"
 К
inputs         d
к "-в*
#К 
tensor_0         А1
Ъ Ж
)__inference_dense_1_layer_call_fn_2926397Y23/в,
%в"
 К
inputs         d
к ""К
unknown         А1▓
H__inference_embedding_2_layer_call_and_return_conditional_losses_2926365f/в,
%в"
 К
inputs         
к "0в-
&К#
tensor_0         d
Ъ М
-__inference_embedding_2_layer_call_fn_2926356[/в,
%в"
 К
inputs         
к "%К"
unknown         dн
F__inference_flatten_3_layer_call_and_return_conditional_losses_2926376c3в0
)в&
$К!
inputs         d
к ",в)
"К
tensor_0         d
Ъ З
+__inference_flatten_3_layer_call_fn_2926370X3в0
)в&
$К!
inputs         d
к "!К
unknown         dп
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_2926417a0в-
&в#
!К
inputs         А1
к "-в*
#К 
tensor_0         А1
Ъ Й
/__inference_leaky_re_lu_2_layer_call_fn_2926412V0в-
&в#
!К
inputs         А1
к ""К
unknown         А1┐
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_2926568q8в5
.в+
)К&
inputs         А
к "5в2
+К(
tensor_0         А
Ъ Щ
/__inference_leaky_re_lu_3_layer_call_fn_2926563f8в5
.в+
)К&
inputs         А
к "*К'
unknown         А╜
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_2926682o7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ Ч
/__inference_leaky_re_lu_4_layer_call_fn_2926677d7в4
-в*
(К%
inputs         @
к ")К&
unknown         @·
D__inference_model_1_layer_call_and_return_conditional_losses_2926096▒23CDABQRabcdkl{|}~ЕЖ`в]
VвS
IЪF
!К
input_3         d
!К
input_4         
p

 
к "4в1
*К'
tensor_0         
Ъ ·
D__inference_model_1_layer_call_and_return_conditional_losses_2926157▒23DACBQRabcdkl{|}~ЕЖ`в]
VвS
IЪF
!К
input_3         d
!К
input_4         
p 

 
к "4в1
*К'
tensor_0         
Ъ ╘
)__inference_model_1_layer_call_fn_2926205ж23CDABQRabcdkl{|}~ЕЖ`в]
VвS
IЪF
!К
input_3         d
!К
input_4         
p

 
к ")К&
unknown         ╘
)__inference_model_1_layer_call_fn_2926253ж23DACBQRabcdkl{|}~ЕЖ`в]
VвS
IЪF
!К
input_3         d
!К
input_4         
p 

 
к ")К&
unknown         ╓
G__inference_multiply_1_layer_call_and_return_conditional_losses_2926388КZвW
PвM
KЪH
"К
inputs_0         d
"К
inputs_1         d
к ",в)
"К
tensor_0         d
Ъ п
,__inference_multiply_1_layer_call_fn_2926382ZвW
PвM
KЪH
"К
inputs_0         d
"К
inputs_1         d
к "!К
unknown         d│
F__inference_reshape_1_layer_call_and_return_conditional_losses_2926516i0в-
&в#
!К
inputs         А1
к "5в2
+К(
tensor_0         А
Ъ Н
+__inference_reshape_1_layer_call_fn_2926502^0в-
&в#
!К
inputs         А1
к "*К'
unknown         А 
%__inference_signature_wrapper_2926349╒23DACBQRabcdkl{|}~ЕЖiвf
в 
_к\
,
input_3!К
input_3         d
,
input_4!К
input_4         "OкL
J
conv2d_transpose_24К1
conv2d_transpose_2         