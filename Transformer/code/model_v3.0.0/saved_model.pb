λΜ
ϊ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
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
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Α
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02unknown8¨

layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelayer_normalization/gamma

-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:*
dtype0

layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namelayer_normalization/beta

,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:*
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0

layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_1/gamma

/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:*
dtype0

layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_1/beta

.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Π
*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	Π
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
	*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:
	*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:	*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
’
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!multi_head_attention/query/kernel

5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
:*
dtype0

multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!multi_head_attention/query/bias

3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:*
dtype0

multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!multi_head_attention/key/kernel

3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
:*
dtype0

multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namemulti_head_attention/key/bias

1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:*
dtype0
’
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!multi_head_attention/value/kernel

5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
:*
dtype0

multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!multi_head_attention/value/bias

3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:*
dtype0
Έ
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,multi_head_attention/attention_output/kernel
±
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:*
dtype0
¬
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*multi_head_attention/attention_output/bias
₯
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

 Adam/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/layer_normalization/gamma/m

4Adam/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/layer_normalization/gamma/m*
_output_shapes
:*
dtype0

Adam/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/layer_normalization/beta/m

3Adam/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/layer_normalization/beta/m*
_output_shapes
:*
dtype0

Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d/kernel/m

(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*"
_output_shapes
:*
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
:*
dtype0

Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_1/kernel/m

*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*"
_output_shapes
:*
dtype0

Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/m
y
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes
:*
dtype0

"Adam/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/layer_normalization_1/gamma/m

6Adam/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_1/gamma/m*
_output_shapes
:*
dtype0

!Adam/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/layer_normalization_1/beta/m

5Adam/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/layer_normalization_1/beta/m*
_output_shapes
:*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Π
*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	Π
*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
	*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:
	*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:	*
dtype0
°
(Adam/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/multi_head_attention/query/kernel/m
©
<Adam/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention/query/kernel/m*"
_output_shapes
:*
dtype0
¨
&Adam/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/multi_head_attention/query/bias/m
‘
:Adam/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/query/bias/m*
_output_shapes

:*
dtype0
¬
&Adam/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/multi_head_attention/key/kernel/m
₯
:Adam/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/key/kernel/m*"
_output_shapes
:*
dtype0
€
$Adam/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/multi_head_attention/key/bias/m

8Adam/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp$Adam/multi_head_attention/key/bias/m*
_output_shapes

:*
dtype0
°
(Adam/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/multi_head_attention/value/kernel/m
©
<Adam/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention/value/kernel/m*"
_output_shapes
:*
dtype0
¨
&Adam/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/multi_head_attention/value/bias/m
‘
:Adam/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/value/bias/m*
_output_shapes

:*
dtype0
Ζ
3Adam/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/multi_head_attention/attention_output/kernel/m
Ώ
GAdam/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp3Adam/multi_head_attention/attention_output/kernel/m*"
_output_shapes
:*
dtype0
Ί
1Adam/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/multi_head_attention/attention_output/bias/m
³
EAdam/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOp1Adam/multi_head_attention/attention_output/bias/m*
_output_shapes
:*
dtype0

 Adam/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/layer_normalization/gamma/v

4Adam/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/layer_normalization/gamma/v*
_output_shapes
:*
dtype0

Adam/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/layer_normalization/beta/v

3Adam/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/layer_normalization/beta/v*
_output_shapes
:*
dtype0

Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d/kernel/v

(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*"
_output_shapes
:*
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
:*
dtype0

Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_1/kernel/v

*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*"
_output_shapes
:*
dtype0

Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/v
y
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes
:*
dtype0

"Adam/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/layer_normalization_1/gamma/v

6Adam/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_1/gamma/v*
_output_shapes
:*
dtype0

!Adam/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/layer_normalization_1/beta/v

5Adam/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/layer_normalization_1/beta/v*
_output_shapes
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Π
*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	Π
*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:
*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
	*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:
	*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:	*
dtype0
°
(Adam/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/multi_head_attention/query/kernel/v
©
<Adam/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention/query/kernel/v*"
_output_shapes
:*
dtype0
¨
&Adam/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/multi_head_attention/query/bias/v
‘
:Adam/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/query/bias/v*
_output_shapes

:*
dtype0
¬
&Adam/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/multi_head_attention/key/kernel/v
₯
:Adam/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/key/kernel/v*"
_output_shapes
:*
dtype0
€
$Adam/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/multi_head_attention/key/bias/v

8Adam/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp$Adam/multi_head_attention/key/bias/v*
_output_shapes

:*
dtype0
°
(Adam/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/multi_head_attention/value/kernel/v
©
<Adam/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention/value/kernel/v*"
_output_shapes
:*
dtype0
¨
&Adam/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/multi_head_attention/value/bias/v
‘
:Adam/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/value/bias/v*
_output_shapes

:*
dtype0
Ζ
3Adam/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/multi_head_attention/attention_output/kernel/v
Ώ
GAdam/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp3Adam/multi_head_attention/attention_output/kernel/v*"
_output_shapes
:*
dtype0
Ί
1Adam/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/multi_head_attention/attention_output/bias/v
³
EAdam/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOp1Adam/multi_head_attention/attention_output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Β
value·B³ B«
Ό
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
ω
_query_dense

_key_dense
_value_dense
_softmax
_dropout_layer
_output_dense
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*

$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
―
*axis
	+gamma
,beta
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses*

3	keras_api* 
¦

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*

<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses* 
¦

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses*
―
Jaxis
	Kgamma
Lbeta
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses*

S	keras_api* 

T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses* 
¦

Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses*

b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses* 
¦

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses*
Τ
piter

qbeta_1

rbeta_2
	sdecay
tlearning_rate+m,m4m5mBmCmKmLmZm[mhmimumvmwmxmymzm {m‘|m’+v£,v€4v₯5v¦Bv§Cv¨Kv©LvͺZv«[v¬hv­iv?uv―vv°wv±xv²yv³zv΄{v΅|vΆ*

u0
v1
w2
x3
y4
z5
{6
|7
+8
,9
410
511
B12
C13
K14
L15
Z16
[17
h18
i19*

u0
v1
w2
x3
y4
z5
{6
|7
+8
,9
410
511
B12
C13
K14
L15
Z16
[17
h18
i19*
* 
²
}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
ί
partial_output_shape
full_output_shape

ukernel
vbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
ί
partial_output_shape
full_output_shape

wkernel
xbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
ί
partial_output_shape
full_output_shape

ykernel
zbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses* 

‘	variables
’trainable_variables
£regularization_losses
€	keras_api
₯__call__
+¦&call_and_return_all_conditional_losses* 
ί
§partial_output_shape
¨full_output_shape

{kernel
|bias
©	variables
ͺtrainable_variables
«regularization_losses
¬	keras_api
­__call__
+?&call_and_return_all_conditional_losses*
<
u0
v1
w2
x3
y4
z5
{6
|7*
<
u0
v1
w2
x3
y4
z5
{6
|7*
* 

―non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

΄non_trainable_variables
΅layers
Άmetrics
 ·layer_regularization_losses
Έlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 
* 
* 
* 
hb
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*
* 

Ήnon_trainable_variables
Ίlayers
»metrics
 Όlayer_regularization_losses
½layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*
* 
* 
* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv1d/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*
* 

Ύnon_trainable_variables
Ώlayers
ΐmetrics
 Αlayer_regularization_losses
Βlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Γnon_trainable_variables
Δlayers
Εmetrics
 Ζlayer_regularization_losses
Ηlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

B0
C1*

B0
C1*
* 

Θnon_trainable_variables
Ιlayers
Κmetrics
 Λlayer_regularization_losses
Μlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*

K0
L1*

K0
L1*
* 

Νnon_trainable_variables
Ξlayers
Οmetrics
 Πlayer_regularization_losses
Ρlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

?non_trainable_variables
Σlayers
Τmetrics
 Υlayer_regularization_losses
Φlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

Z0
[1*

Z0
[1*
* 

Χnon_trainable_variables
Ψlayers
Ωmetrics
 Ϊlayer_regularization_losses
Ϋlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

άnon_trainable_variables
έlayers
ήmetrics
 ίlayer_regularization_losses
ΰlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

h0
i1*

h0
i1*
* 

αnon_trainable_variables
βlayers
γmetrics
 δlayer_regularization_losses
εlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention/query/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/query/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/key/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmulti_head_attention/key/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention/value/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/value/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,multi_head_attention/attention_output/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*multi_head_attention/attention_output/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
j
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
13*

ζ0
η1*
* 
* 
* 
* 
* 

u0
v1*

u0
v1*
* 

θnon_trainable_variables
ιlayers
κmetrics
 λlayer_regularization_losses
μlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 

w0
x1*

w0
x1*
* 

νnon_trainable_variables
ξlayers
οmetrics
 πlayer_regularization_losses
ρlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 

y0
z1*

y0
z1*
* 

ςnon_trainable_variables
σlayers
τmetrics
 υlayer_regularization_losses
φlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

χnon_trainable_variables
ψlayers
ωmetrics
 ϊlayer_regularization_losses
ϋlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

όnon_trainable_variables
ύlayers
ώmetrics
 ?layer_regularization_losses
layer_metrics
‘	variables
’trainable_variables
£regularization_losses
₯__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 
* 
* 
* 
* 

{0
|1*

{0
|1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
©	variables
ͺtrainable_variables
«regularization_losses
­__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
.
0
1
2
3
4
5*
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
<

total

count
	variables
	keras_api*
M

total

count

_fn_kwargs
	variables
	keras_api*
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
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*

VARIABLE_VALUE Adam/layer_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/layer_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv1d/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv1d/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv1d_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/layer_normalization_1/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/layer_normalization_1/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/multi_head_attention/query/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/multi_head_attention/query/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/multi_head_attention/key/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/multi_head_attention/key/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/multi_head_attention/value/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/multi_head_attention/value/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/multi_head_attention/attention_output/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE1Adam/multi_head_attention/attention_output/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/layer_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/layer_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv1d/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv1d/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv1d_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/layer_normalization_1/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/layer_normalization_1/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/multi_head_attention/query/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/multi_head_attention/query/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/multi_head_attention/key/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/multi_head_attention/key/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/multi_head_attention/value/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/multi_head_attention/value/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/multi_head_attention/attention_output/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE1Adam/multi_head_attention/attention_output/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*,
_output_shapes
:?????????Π*
dtype0*!
shape:?????????Π
α
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biaslayer_normalization/gammalayer_normalization/betaconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biaslayer_normalization_1/gammalayer_normalization_1/betadense/kernel
dense/biasdense_1/kerneldense_1/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_501721
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Π
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp5multi_head_attention/query/kernel/Read/ReadVariableOp3multi_head_attention/query/bias/Read/ReadVariableOp3multi_head_attention/key/kernel/Read/ReadVariableOp1multi_head_attention/key/bias/Read/ReadVariableOp5multi_head_attention/value/kernel/Read/ReadVariableOp3multi_head_attention/value/bias/Read/ReadVariableOp@multi_head_attention/attention_output/kernel/Read/ReadVariableOp>multi_head_attention/attention_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp4Adam/layer_normalization/gamma/m/Read/ReadVariableOp3Adam/layer_normalization/beta/m/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp6Adam/layer_normalization_1/gamma/m/Read/ReadVariableOp5Adam/layer_normalization_1/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp<Adam/multi_head_attention/query/kernel/m/Read/ReadVariableOp:Adam/multi_head_attention/query/bias/m/Read/ReadVariableOp:Adam/multi_head_attention/key/kernel/m/Read/ReadVariableOp8Adam/multi_head_attention/key/bias/m/Read/ReadVariableOp<Adam/multi_head_attention/value/kernel/m/Read/ReadVariableOp:Adam/multi_head_attention/value/bias/m/Read/ReadVariableOpGAdam/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpEAdam/multi_head_attention/attention_output/bias/m/Read/ReadVariableOp4Adam/layer_normalization/gamma/v/Read/ReadVariableOp3Adam/layer_normalization/beta/v/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp6Adam/layer_normalization_1/gamma/v/Read/ReadVariableOp5Adam/layer_normalization_1/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp<Adam/multi_head_attention/query/kernel/v/Read/ReadVariableOp:Adam/multi_head_attention/query/bias/v/Read/ReadVariableOp:Adam/multi_head_attention/key/kernel/v/Read/ReadVariableOp8Adam/multi_head_attention/key/bias/v/Read/ReadVariableOp<Adam/multi_head_attention/value/kernel/v/Read/ReadVariableOp:Adam/multi_head_attention/value/bias/v/Read/ReadVariableOpGAdam/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpEAdam/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpConst*R
TinK
I2G	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_502315
η
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betaconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biaslayer_normalization_1/gammalayer_normalization_1/betadense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biastotalcounttotal_1count_1 Adam/layer_normalization/gamma/mAdam/layer_normalization/beta/mAdam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/m"Adam/layer_normalization_1/gamma/m!Adam/layer_normalization_1/beta/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/m(Adam/multi_head_attention/query/kernel/m&Adam/multi_head_attention/query/bias/m&Adam/multi_head_attention/key/kernel/m$Adam/multi_head_attention/key/bias/m(Adam/multi_head_attention/value/kernel/m&Adam/multi_head_attention/value/bias/m3Adam/multi_head_attention/attention_output/kernel/m1Adam/multi_head_attention/attention_output/bias/m Adam/layer_normalization/gamma/vAdam/layer_normalization/beta/vAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/v"Adam/layer_normalization_1/gamma/v!Adam/layer_normalization_1/beta/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v(Adam/multi_head_attention/query/kernel/v&Adam/multi_head_attention/query/bias/v&Adam/multi_head_attention/key/kernel/v$Adam/multi_head_attention/key/bias/v(Adam/multi_head_attention/value/kernel/v&Adam/multi_head_attention/value/bias/v3Adam/multi_head_attention/attention_output/kernel/v1Adam/multi_head_attention/attention_output/bias/v*Q
TinJ
H2F*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_502532Α³
²
D
(__inference_dropout_layer_call_fn_501847

inputs
identityΆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_500604e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????Π"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
ο
 
&__inference_model_layer_call_fn_501208
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:	Π


unknown_16:


unknown_17:
	

unknown_18:	
identity’StatefulPartitionedCallΝ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_501120o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????Π
!
_user_specified_name	input_1
μ

&__inference_model_layer_call_fn_501418

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:	Π


unknown_16:


unknown_17:
	

unknown_18:	
identity’StatefulPartitionedCallΜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_501120o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
χ

D__inference_conv1d_1_layer_call_and_return_conditional_losses_500679

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Π
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Π*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Π*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????Π
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs


d
E__inference_dropout_1_layer_call_and_return_conditional_losses_500883

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΠC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Π*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Πt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Πn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Π^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Π"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs


τ
C__inference_dense_1_layer_call_and_return_conditional_losses_500750

inputs0
matmul_readvariableop_resource:
	-
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????	`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
>


A__inference_model_layer_call_and_return_conditional_losses_501322
input_11
multi_head_attention_501268:-
multi_head_attention_501270:1
multi_head_attention_501272:-
multi_head_attention_501274:1
multi_head_attention_501276:-
multi_head_attention_501278:1
multi_head_attention_501280:)
multi_head_attention_501282:(
layer_normalization_501286:(
layer_normalization_501288:#
conv1d_501292:
conv1d_501294:%
conv1d_1_501298:
conv1d_1_501300:*
layer_normalization_1_501303:*
layer_normalization_1_501305:
dense_501310:	Π

dense_501312:
 
dense_1_501316:
	
dense_1_501318:	
identity’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dropout/StatefulPartitionedCall’!dropout_1/StatefulPartitionedCall’!dropout_2/StatefulPartitionedCall’+layer_normalization/StatefulPartitionedCall’-layer_normalization_1/StatefulPartitionedCall’,multi_head_attention/StatefulPartitionedCallν
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinput_1input_1multi_head_attention_501268multi_head_attention_501270multi_head_attention_501272multi_head_attention_501274multi_head_attention_501276multi_head_attention_501278multi_head_attention_501280multi_head_attention_501282*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_500997ύ
dropout/StatefulPartitionedCallStatefulPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_500926Ζ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0layer_normalization_501286layer_normalization_501288*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_500628
tf.__operators__.add/AddV2AddV24layer_normalization/StatefulPartitionedCall:output:0input_1*
T0*,
_output_shapes
:?????????Π
conv1d/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_501292conv1d_501294*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_500651
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_500883
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_1_501298conv1d_1_501300*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_500679Ο
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0layer_normalization_1_501303layer_normalization_1_501305*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_500707΄
tf.__operators__.add_1/AddV2AddV26layer_normalization_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Πφ
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_500536
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_501310dense_501312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_500726
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_500830
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_1_501316dense_1_501318*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_500750w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	Γ
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall:U Q
,
_output_shapes
:?????????Π
!
_user_specified_name	input_1
ο

6__inference_layer_normalization_1_layer_call_fn_501985

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallξ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_500707t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Π`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
²*
ω
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_501800	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Π
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Π
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ΠJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5?d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Π
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΠΠ*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΠΠs
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΠΠ¦
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Π*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Π*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΠΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π:?????????Π: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Π

_user_specified_namequery:SO
,
_output_shapes
:?????????Π

_user_specified_namevalue
χ

D__inference_conv1d_1_layer_call_and_return_conditional_losses_501976

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Π
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Π*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Π*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????Π
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs


τ
C__inference_dense_1_layer_call_and_return_conditional_losses_502085

inputs0
matmul_readvariableop_resource:
	-
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????	`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
δΑ

A__inference_model_layer_call_and_return_conditional_losses_501674

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:H
2conv1d_conv1d_expanddims_1_readvariableop_resource:4
&conv1d_biasadd_readvariableop_resource:J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	Π
3
%dense_biasadd_readvariableop_resource:
8
&dense_1_matmul_readvariableop_resource:
	5
'dense_1_biasadd_readvariableop_resource:	
identity’conv1d/BiasAdd/ReadVariableOp’)conv1d/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_1/BiasAdd/ReadVariableOp’+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’,layer_normalization/batchnorm/ReadVariableOp’0layer_normalization/batchnorm/mul/ReadVariableOp’.layer_normalization_1/batchnorm/ReadVariableOp’2layer_normalization_1/batchnorm/mul/ReadVariableOp’8multi_head_attention/attention_output/add/ReadVariableOp’Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp’+multi_head_attention/key/add/ReadVariableOp’5multi_head_attention/key/einsum/Einsum/ReadVariableOp’-multi_head_attention/query/add/ReadVariableOp’7multi_head_attention/query/einsum/Einsum/ReadVariableOp’-multi_head_attention/value/add/ReadVariableOp’7multi_head_attention/value/einsum/Einsum/ReadVariableOpΌ
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ά
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abde€
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0Μ
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ΠΈ
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ψ
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0Ζ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ΠΌ
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ά
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abde€
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0Μ
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Π_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5?£
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:?????????ΠΠ
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΠΠ*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΠΠo
*multi_head_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?Π
(multi_head_attention/dropout/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:03multi_head_attention/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΠΠ
*multi_head_attention/dropout/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:Π
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform3multi_head_attention/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΠΠ*
dtype0x
3multi_head_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=
1multi_head_attention/dropout/dropout/GreaterEqualGreaterEqualJmulti_head_attention/dropout/dropout/random_uniform/RandomUniform:output:0<multi_head_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΠΠ³
)multi_head_attention/dropout/dropout/CastCast5multi_head_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΠΠΚ
*multi_head_attention/dropout/dropout/Mul_1Mul,multi_head_attention/dropout/dropout/Mul:z:0-multi_head_attention/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΠΠε
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Π*
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Π*
equationabcd,cde->abeΆ
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ι
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΠZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8? 
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:?????????Πr
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:‘
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????Π*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=Γ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Π
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Π
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Π|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Θ
 layer_normalization/moments/meanMeandropout/dropout/Mul_1:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:?????????ΠΗ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout/dropout/Mul_1:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Π
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:θ
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ύ
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Π
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Π¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Β
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Π£
#layer_normalization/batchnorm/mul_1Muldropout/dropout/Mul_1:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Π³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Π
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ύ
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Π³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Π
tf.__operators__.add/AddV2AddV2'layer_normalization/batchnorm/add_1:z:0inputs*
T0*,
_output_shapes
:?????????Πg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????¨
conv1d/Conv1D/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Π 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ΅
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Γ
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Π*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????Π*
squeeze_dims

ύ????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πc
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Π\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?
dropout_1/dropout/MulMulconv1d/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:?????????Π`
dropout_1/dropout/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:₯
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????Π*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=Ι
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Π
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Π
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Πi
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????©
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_1/dropout/Mul_1:z:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Π€
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Π*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????Π*
squeeze_dims

ύ????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Π~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Μ
"layer_normalization_1/moments/meanMeanconv1d_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:?????????ΠΛ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Π
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Π
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Πͺ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Π§
%layer_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΠΉ
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Π’
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΠΉ
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Π§
tf.__operators__.add_1/AddV2AddV2)layer_normalization_1/batchnorm/add_1:z:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Πq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :΄
global_average_pooling1d/MeanMean tf.__operators__.add_1/AddV2:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:?????????Π
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	Π
*
dtype0
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?
dropout_2/dropout/MulMuldense/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????
_
dropout_2/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=Δ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????

dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????

dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????

dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
	*
dtype0
dense_1/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	β
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
ώ

Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_502007

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Π
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Πl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Πb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Π~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Πw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Πv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Πw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Πg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Π
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
έ

)__inference_conv1d_1_layer_call_fn_501961

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallα
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_500679t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Π`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
ώ
U
9__inference_global_average_pooling1d_layer_call_fn_502012

inputs
identityΛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_500536i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

κ
5__inference_multi_head_attention_layer_call_fn_501743	
query	
value
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity’StatefulPartitionedCallΒ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_500581t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Π`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π:?????????Π: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????Π

_user_specified_namequery:SO
,
_output_shapes
:?????????Π

_user_specified_namevalue
α9
¨	
A__inference_model_layer_call_and_return_conditional_losses_500757

inputs1
multi_head_attention_500582:-
multi_head_attention_500584:1
multi_head_attention_500586:-
multi_head_attention_500588:1
multi_head_attention_500590:-
multi_head_attention_500592:1
multi_head_attention_500594:)
multi_head_attention_500596:(
layer_normalization_500629:(
layer_normalization_500631:#
conv1d_500652:
conv1d_500654:%
conv1d_1_500680:
conv1d_1_500682:*
layer_normalization_1_500708:*
layer_normalization_1_500710:
dense_500727:	Π

dense_500729:
 
dense_1_500751:
	
dense_1_500753:	
identity’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’+layer_normalization/StatefulPartitionedCall’-layer_normalization_1/StatefulPartitionedCall’,multi_head_attention/StatefulPartitionedCallλ
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinputsinputsmulti_head_attention_500582multi_head_attention_500584multi_head_attention_500586multi_head_attention_500588multi_head_attention_500590multi_head_attention_500592multi_head_attention_500594multi_head_attention_500596*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_500581ν
dropout/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_500604Ύ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0layer_normalization_500629layer_normalization_500631*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_500628
tf.__operators__.add/AddV2AddV24layer_normalization/StatefulPartitionedCall:output:0inputs*
T0*,
_output_shapes
:?????????Π
conv1d/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_500652conv1d_500654*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_500651γ
dropout_1/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_500662
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_1_500680conv1d_1_500682*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_500679Ο
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0layer_normalization_1_500708layer_normalization_1_500710*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_500707΄
tf.__operators__.add_1/AddV2AddV26layer_normalization_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Πφ
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_500536
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_500727dense_500729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_500726έ
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_500737
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_1_500751dense_1_500753*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_500750w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	Ω
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
κ
a
C__inference_dropout_layer_call_and_return_conditional_losses_500604

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Π`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Π"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs


σ
A__inference_dense_layer_call_and_return_conditional_losses_502038

inputs1
matmul_readvariableop_resource:	Π
-
biasadd_readvariableop_resource:

identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Π
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????Π
 
_user_specified_nameinputs


b
C__inference_dropout_layer_call_and_return_conditional_losses_501869

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΠC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Π*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Πt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Πn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Π^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Π"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
κ
a
C__inference_dropout_layer_call_and_return_conditional_losses_501857

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Π`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Π"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
Ν

$__inference_signature_wrapper_501721
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:	Π


unknown_16:


unknown_17:
	

unknown_18:	
identity’StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_500526o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????Π
!
_user_specified_name	input_1
―
ζ
!__inference__wrapped_model_500526
input_1\
Fmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource:N
<model_multi_head_attention_query_add_readvariableop_resource:Z
Dmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource:L
:model_multi_head_attention_key_add_readvariableop_resource:\
Fmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource:N
<model_multi_head_attention_value_add_readvariableop_resource:g
Qmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:U
Gmodel_multi_head_attention_attention_output_add_readvariableop_resource:M
?model_layer_normalization_batchnorm_mul_readvariableop_resource:I
;model_layer_normalization_batchnorm_readvariableop_resource:N
8model_conv1d_conv1d_expanddims_1_readvariableop_resource::
,model_conv1d_biasadd_readvariableop_resource:P
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_1_biasadd_readvariableop_resource:O
Amodel_layer_normalization_1_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_1_batchnorm_readvariableop_resource:=
*model_dense_matmul_readvariableop_resource:	Π
9
+model_dense_biasadd_readvariableop_resource:
>
,model_dense_1_matmul_readvariableop_resource:
	;
-model_dense_1_biasadd_readvariableop_resource:	
identity’#model/conv1d/BiasAdd/ReadVariableOp’/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp’%model/conv1d_1/BiasAdd/ReadVariableOp’1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp’"model/dense/BiasAdd/ReadVariableOp’!model/dense/MatMul/ReadVariableOp’$model/dense_1/BiasAdd/ReadVariableOp’#model/dense_1/MatMul/ReadVariableOp’2model/layer_normalization/batchnorm/ReadVariableOp’6model/layer_normalization/batchnorm/mul/ReadVariableOp’4model/layer_normalization_1/batchnorm/ReadVariableOp’8model/layer_normalization_1/batchnorm/mul/ReadVariableOp’>model/multi_head_attention/attention_output/add/ReadVariableOp’Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp’1model/multi_head_attention/key/add/ReadVariableOp’;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp’3model/multi_head_attention/query/add/ReadVariableOp’=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp’3model/multi_head_attention/value/add/ReadVariableOp’=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpΘ
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ι
.model/multi_head_attention/query/einsum/EinsumEinsuminput_1Emodel/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abde°
3model/multi_head_attention/query/add/ReadVariableOpReadVariableOp<model_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0ή
$model/multi_head_attention/query/addAddV27model/multi_head_attention/query/einsum/Einsum:output:0;model/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ΠΔ
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpDmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ε
,model/multi_head_attention/key/einsum/EinsumEinsuminput_1Cmodel/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abde¬
1model/multi_head_attention/key/add/ReadVariableOpReadVariableOp:model_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0Ψ
"model/multi_head_attention/key/addAddV25model/multi_head_attention/key/einsum/Einsum:output:09model/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ΠΘ
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ι
.model/multi_head_attention/value/einsum/EinsumEinsuminput_1Emodel/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abde°
3model/multi_head_attention/value/add/ReadVariableOpReadVariableOp<model_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0ή
$model/multi_head_attention/value/addAddV27model/multi_head_attention/value/einsum/Einsum:output:0;model/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Πe
 model/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5?΅
model/multi_head_attention/MulMul(model/multi_head_attention/query/add:z:0)model/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:?????????Πβ
(model/multi_head_attention/einsum/EinsumEinsum&model/multi_head_attention/key/add:z:0"model/multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΠΠ*
equationaecd,abcd->acbe€
*model/multi_head_attention/softmax/SoftmaxSoftmax1model/multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΠΠ©
+model/multi_head_attention/dropout/IdentityIdentity4model/multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΠΠχ
*model/multi_head_attention/einsum_1/EinsumEinsum4model/multi_head_attention/dropout/Identity:output:0(model/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Π*
equationacbe,aecd->abcdή
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0§
9model/multi_head_attention/attention_output/einsum/EinsumEinsum3model/multi_head_attention/einsum_1/Einsum:output:0Pmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Π*
equationabcd,cde->abeΒ
>model/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpGmodel_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ϋ
/model/multi_head_attention/attention_output/addAddV2Bmodel/multi_head_attention/attention_output/einsum/Einsum:output:0Fmodel/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Π
model/dropout/IdentityIdentity3model/multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:?????????Π
8model/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ϊ
&model/layer_normalization/moments/meanMeanmodel/dropout/Identity:output:0Amodel/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(¦
.model/layer_normalization/moments/StopGradientStopGradient/model/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:?????????ΠΩ
3model/layer_normalization/moments/SquaredDifferenceSquaredDifferencemodel/dropout/Identity:output:07model/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Π
<model/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ϊ
*model/layer_normalization/moments/varianceMean7model/layer_normalization/moments/SquaredDifference:z:0Emodel/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(n
)model/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Π
'model/layer_normalization/batchnorm/addAddV23model/layer_normalization/moments/variance:output:02model/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Π
)model/layer_normalization/batchnorm/RsqrtRsqrt+model/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Π²
6model/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Τ
'model/layer_normalization/batchnorm/mulMul-model/layer_normalization/batchnorm/Rsqrt:y:0>model/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Π΅
)model/layer_normalization/batchnorm/mul_1Mulmodel/dropout/Identity:output:0+model/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΠΕ
)model/layer_normalization/batchnorm/mul_2Mul/model/layer_normalization/moments/mean:output:0+model/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Πͺ
2model/layer_normalization/batchnorm/ReadVariableOpReadVariableOp;model_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Π
'model/layer_normalization/batchnorm/subSub:model/layer_normalization/batchnorm/ReadVariableOp:value:0-model/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΠΕ
)model/layer_normalization/batchnorm/add_1AddV2-model/layer_normalization/batchnorm/mul_1:z:0+model/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Π
 model/tf.__operators__.add/AddV2AddV2-model/layer_normalization/batchnorm/add_1:z:0input_1*
T0*,
_output_shapes
:?????????Πm
"model/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????Ί
model/conv1d/Conv1D/ExpandDims
ExpandDims$model/tf.__operators__.add/AddV2:z:0+model/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Π¬
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0f
$model/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Η
 model/conv1d/Conv1D/ExpandDims_1
ExpandDims7model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Υ
model/conv1d/Conv1DConv2D'model/conv1d/Conv1D/ExpandDims:output:0)model/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Π*
paddingVALID*
strides

model/conv1d/Conv1D/SqueezeSqueezemodel/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????Π*
squeeze_dims

ύ????????
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model/conv1d/BiasAddBiasAdd$model/conv1d/Conv1D/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πo
model/conv1d/ReluRelumodel/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Π|
model/dropout_1/IdentityIdentitymodel/conv1d/Relu:activations:0*
T0*,
_output_shapes
:?????????Πo
$model/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????»
 model/conv1d_1/Conv1D/ExpandDims
ExpandDims!model/dropout_1/Identity:output:0-model/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Π°
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ν
"model/conv1d_1/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ϋ
model/conv1d_1/Conv1DConv2D)model/conv1d_1/Conv1D/ExpandDims:output:0+model/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Π*
paddingVALID*
strides

model/conv1d_1/Conv1D/SqueezeSqueezemodel/conv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????Π*
squeeze_dims

ύ????????
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0―
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/Conv1D/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Π
:model/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ή
(model/layer_normalization_1/moments/meanMeanmodel/conv1d_1/BiasAdd:output:0Cmodel/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(ͺ
0model/layer_normalization_1/moments/StopGradientStopGradient1model/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:?????????Πέ
5model/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencemodel/conv1d_1/BiasAdd:output:09model/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Π
>model/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
,model/layer_normalization_1/moments/varianceMean9model/layer_normalization_1/moments/SquaredDifference:z:0Gmodel/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(p
+model/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Φ
)model/layer_normalization_1/batchnorm/addAddV25model/layer_normalization_1/moments/variance:output:04model/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Π
+model/layer_normalization_1/batchnorm/RsqrtRsqrt-model/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????ΠΆ
8model/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ϊ
)model/layer_normalization_1/batchnorm/mulMul/model/layer_normalization_1/batchnorm/Rsqrt:y:0@model/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΠΉ
+model/layer_normalization_1/batchnorm/mul_1Mulmodel/conv1d_1/BiasAdd:output:0-model/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΠΛ
+model/layer_normalization_1/batchnorm/mul_2Mul1model/layer_normalization_1/moments/mean:output:0-model/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Π?
4model/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Φ
)model/layer_normalization_1/batchnorm/subSub<model/layer_normalization_1/batchnorm/ReadVariableOp:value:0/model/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΠΛ
+model/layer_normalization_1/batchnorm/add_1AddV2/model/layer_normalization_1/batchnorm/mul_1:z:0-model/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????ΠΉ
"model/tf.__operators__.add_1/AddV2AddV2/model/layer_normalization_1/batchnorm/add_1:z:0$model/tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Πw
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ζ
#model/global_average_pooling1d/MeanMean&model/tf.__operators__.add_1/AddV2:z:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:?????????Π
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	Π
*
dtype0§
model/dense/MatMulMatMul,model/global_average_pooling1d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????

"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
v
model/dropout_2/IdentityIdentitymodel/dense/Relu:activations:0*
T0*'
_output_shapes
:?????????

#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:
	*
dtype0 
model/dense_1/MatMulMatMul!model/dropout_2/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0 
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r
model/dense_1/SoftmaxSoftmaxmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	n
IdentityIdentitymodel/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	Ϊ
NoOpNoOp$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp3^model/layer_normalization/batchnorm/ReadVariableOp7^model/layer_normalization/batchnorm/mul/ReadVariableOp5^model/layer_normalization_1/batchnorm/ReadVariableOp9^model/layer_normalization_1/batchnorm/mul/ReadVariableOp?^model/multi_head_attention/attention_output/add/ReadVariableOpI^model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2^model/multi_head_attention/key/add/ReadVariableOp<^model/multi_head_attention/key/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/query/add/ReadVariableOp>^model/multi_head_attention/query/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/value/add/ReadVariableOp>^model/multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 2J
#model/conv1d/BiasAdd/ReadVariableOp#model/conv1d/BiasAdd/ReadVariableOp2b
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_1/BiasAdd/ReadVariableOp%model/conv1d_1/BiasAdd/ReadVariableOp2f
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2h
2model/layer_normalization/batchnorm/ReadVariableOp2model/layer_normalization/batchnorm/ReadVariableOp2p
6model/layer_normalization/batchnorm/mul/ReadVariableOp6model/layer_normalization/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_1/batchnorm/ReadVariableOp4model/layer_normalization_1/batchnorm/ReadVariableOp2t
8model/layer_normalization_1/batchnorm/mul/ReadVariableOp8model/layer_normalization_1/batchnorm/mul/ReadVariableOp2
>model/multi_head_attention/attention_output/add/ReadVariableOp>model/multi_head_attention/attention_output/add/ReadVariableOp2
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2f
1model/multi_head_attention/key/add/ReadVariableOp1model/multi_head_attention/key/add/ReadVariableOp2z
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/query/add/ReadVariableOp3model/multi_head_attention/query/add/ReadVariableOp2~
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/value/add/ReadVariableOp3model/multi_head_attention/value/add/ReadVariableOp2~
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp:U Q
,
_output_shapes
:?????????Π
!
_user_specified_name	input_1
Ω

'__inference_conv1d_layer_call_fn_501909

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallί
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_500651t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Π`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
’
F
*__inference_dropout_2_layer_call_fn_502043

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_500737`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs

c
*__inference_dropout_1_layer_call_fn_501935

inputs
identity’StatefulPartitionedCallΘ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_500883t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Π`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs

a
(__inference_dropout_layer_call_fn_501852

inputs
identity’StatefulPartitionedCallΖ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_500926t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Π`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
Ν
Π 
__inference__traced_save_502315
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop@
<savev2_multi_head_attention_query_kernel_read_readvariableop>
:savev2_multi_head_attention_query_bias_read_readvariableop>
:savev2_multi_head_attention_key_kernel_read_readvariableop<
8savev2_multi_head_attention_key_bias_read_readvariableop@
<savev2_multi_head_attention_value_kernel_read_readvariableop>
:savev2_multi_head_attention_value_bias_read_readvariableopK
Gsavev2_multi_head_attention_attention_output_kernel_read_readvariableopI
Esavev2_multi_head_attention_attention_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop?
;savev2_adam_layer_normalization_gamma_m_read_readvariableop>
:savev2_adam_layer_normalization_beta_m_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableopA
=savev2_adam_layer_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_layer_normalization_1_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopG
Csavev2_adam_multi_head_attention_query_kernel_m_read_readvariableopE
Asavev2_adam_multi_head_attention_query_bias_m_read_readvariableopE
Asavev2_adam_multi_head_attention_key_kernel_m_read_readvariableopC
?savev2_adam_multi_head_attention_key_bias_m_read_readvariableopG
Csavev2_adam_multi_head_attention_value_kernel_m_read_readvariableopE
Asavev2_adam_multi_head_attention_value_bias_m_read_readvariableopR
Nsavev2_adam_multi_head_attention_attention_output_kernel_m_read_readvariableopP
Lsavev2_adam_multi_head_attention_attention_output_bias_m_read_readvariableop?
;savev2_adam_layer_normalization_gamma_v_read_readvariableop>
:savev2_adam_layer_normalization_beta_v_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableopA
=savev2_adam_layer_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_layer_normalization_1_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopG
Csavev2_adam_multi_head_attention_query_kernel_v_read_readvariableopE
Asavev2_adam_multi_head_attention_query_bias_v_read_readvariableopE
Asavev2_adam_multi_head_attention_key_kernel_v_read_readvariableopC
?savev2_adam_multi_head_attention_key_bias_v_read_readvariableopG
Csavev2_adam_multi_head_attention_value_kernel_v_read_readvariableopE
Asavev2_adam_multi_head_attention_value_bias_v_read_readvariableopR
Nsavev2_adam_multi_head_attention_attention_output_kernel_v_read_readvariableopP
Lsavev2_adam_multi_head_attention_attention_output_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: $
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*Ζ#
valueΌ#BΉ#FB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHό
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*‘
valueBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ε
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop<savev2_multi_head_attention_query_kernel_read_readvariableop:savev2_multi_head_attention_query_bias_read_readvariableop:savev2_multi_head_attention_key_kernel_read_readvariableop8savev2_multi_head_attention_key_bias_read_readvariableop<savev2_multi_head_attention_value_kernel_read_readvariableop:savev2_multi_head_attention_value_bias_read_readvariableopGsavev2_multi_head_attention_attention_output_kernel_read_readvariableopEsavev2_multi_head_attention_attention_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop;savev2_adam_layer_normalization_gamma_m_read_readvariableop:savev2_adam_layer_normalization_beta_m_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop=savev2_adam_layer_normalization_1_gamma_m_read_readvariableop<savev2_adam_layer_normalization_1_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableopCsavev2_adam_multi_head_attention_query_kernel_m_read_readvariableopAsavev2_adam_multi_head_attention_query_bias_m_read_readvariableopAsavev2_adam_multi_head_attention_key_kernel_m_read_readvariableop?savev2_adam_multi_head_attention_key_bias_m_read_readvariableopCsavev2_adam_multi_head_attention_value_kernel_m_read_readvariableopAsavev2_adam_multi_head_attention_value_bias_m_read_readvariableopNsavev2_adam_multi_head_attention_attention_output_kernel_m_read_readvariableopLsavev2_adam_multi_head_attention_attention_output_bias_m_read_readvariableop;savev2_adam_layer_normalization_gamma_v_read_readvariableop:savev2_adam_layer_normalization_beta_v_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop=savev2_adam_layer_normalization_1_gamma_v_read_readvariableop<savev2_adam_layer_normalization_1_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopCsavev2_adam_multi_head_attention_query_kernel_v_read_readvariableopAsavev2_adam_multi_head_attention_query_bias_v_read_readvariableopAsavev2_adam_multi_head_attention_key_kernel_v_read_readvariableop?savev2_adam_multi_head_attention_key_bias_v_read_readvariableopCsavev2_adam_multi_head_attention_value_kernel_v_read_readvariableopAsavev2_adam_multi_head_attention_value_bias_v_read_readvariableopNsavev2_adam_multi_head_attention_attention_output_kernel_v_read_readvariableopLsavev2_adam_multi_head_attention_attention_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *T
dtypesJ
H2F	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*β
_input_shapesΠ
Ν: :::::::::	Π
:
:
	:	: : : : : ::::::::: : : : :::::::::	Π
:
:
	:	:::::::::::::::::	Π
:
:
	:	::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%	!

_output_shapes
:	Π
: 


_output_shapes
:
:$ 

_output_shapes

:
	: 

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::( $
"
_output_shapes
:: !

_output_shapes
::("$
"
_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
::%&!

_output_shapes
:	Π
: '

_output_shapes
:
:$( 

_output_shapes

:
	: )

_output_shapes
:	:(*$
"
_output_shapes
::$+ 

_output_shapes

::(,$
"
_output_shapes
::$- 

_output_shapes

::(.$
"
_output_shapes
::$/ 

_output_shapes

::(0$
"
_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
::(4$
"
_output_shapes
:: 5

_output_shapes
::(6$
"
_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
::%:!

_output_shapes
:	Π
: ;

_output_shapes
:
:$< 

_output_shapes

:
	: =

_output_shapes
:	:(>$
"
_output_shapes
::$? 

_output_shapes

::(@$
"
_output_shapes
::$A 

_output_shapes

::(B$
"
_output_shapes
::$C 

_output_shapes

::(D$
"
_output_shapes
:: E

_output_shapes
::F

_output_shapes
: 
ό

O__inference_layer_normalization_layer_call_and_return_conditional_losses_501900

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Π
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Πl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Πb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Π~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Πw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Πv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Πw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Πg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Π
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs

p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_502018

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs


b
C__inference_dropout_layer_call_and_return_conditional_losses_500926

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΠC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Π*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Πt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Πn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Π^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Π"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
Β

&__inference_dense_layer_call_fn_502027

inputs
unknown:	Π

	unknown_0:

identity’StatefulPartitionedCallΩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_500726o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????Π
 
_user_specified_nameinputs


d
E__inference_dropout_1_layer_call_and_return_conditional_losses_501952

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΠC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Π*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Πt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Πn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Π^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Π"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
μ
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_500662

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Π`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Π"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
2
ω
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_500997	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Π
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Π
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ΠJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5?d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Π
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΠΠ*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΠΠZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΠΠ^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:¦
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΠΠ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=Θ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΠΠ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΠΠ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΠΠ¦
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Π*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Π*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΠΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π:?????????Π: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Π

_user_specified_namequery:SO
,
_output_shapes
:?????????Π

_user_specified_namevalue
χ 

A__inference_model_layer_call_and_return_conditional_losses_501532

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:H
2conv1d_conv1d_expanddims_1_readvariableop_resource:4
&conv1d_biasadd_readvariableop_resource:J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	Π
3
%dense_biasadd_readvariableop_resource:
8
&dense_1_matmul_readvariableop_resource:
	5
'dense_1_biasadd_readvariableop_resource:	
identity’conv1d/BiasAdd/ReadVariableOp’)conv1d/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_1/BiasAdd/ReadVariableOp’+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’,layer_normalization/batchnorm/ReadVariableOp’0layer_normalization/batchnorm/mul/ReadVariableOp’.layer_normalization_1/batchnorm/ReadVariableOp’2layer_normalization_1/batchnorm/mul/ReadVariableOp’8multi_head_attention/attention_output/add/ReadVariableOp’Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp’+multi_head_attention/key/add/ReadVariableOp’5multi_head_attention/key/einsum/Einsum/ReadVariableOp’-multi_head_attention/query/add/ReadVariableOp’7multi_head_attention/query/einsum/Einsum/ReadVariableOp’-multi_head_attention/value/add/ReadVariableOp’7multi_head_attention/value/einsum/Einsum/ReadVariableOpΌ
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ά
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abde€
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0Μ
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ΠΈ
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ψ
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0Ζ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ΠΌ
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ά
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abde€
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0Μ
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Π_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5?£
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:?????????ΠΠ
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΠΠ*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΠΠ
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΠΠε
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Π*
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Π*
equationabcd,cde->abeΆ
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ι
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Π
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:?????????Π|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Θ
 layer_normalization/moments/meanMeandropout/Identity:output:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:?????????ΠΗ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout/Identity:output:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Π
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:θ
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ύ
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Π
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Π¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Β
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Π£
#layer_normalization/batchnorm/mul_1Muldropout/Identity:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Π³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Π
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ύ
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Π³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Π
tf.__operators__.add/AddV2AddV2'layer_normalization/batchnorm/add_1:z:0inputs*
T0*,
_output_shapes
:?????????Πg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????¨
conv1d/Conv1D/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Π 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ΅
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Γ
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Π*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????Π*
squeeze_dims

ύ????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πc
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Πp
dropout_1/IdentityIdentityconv1d/Relu:activations:0*
T0*,
_output_shapes
:?????????Πi
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????©
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_1/Identity:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Π€
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Π*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????Π*
squeeze_dims

ύ????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Π~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Μ
"layer_normalization_1/moments/meanMeanconv1d_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:?????????ΠΛ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Π
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Π
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Πͺ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Π§
%layer_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΠΉ
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Π’
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΠΉ
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Π§
tf.__operators__.add_1/AddV2AddV2)layer_normalization_1/batchnorm/add_1:z:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Πq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :΄
global_average_pooling1d/MeanMean tf.__operators__.add_1/AddV2:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:?????????Π
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	Π
*
dtype0
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
j
dropout_2/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:?????????

dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
	*
dtype0
dense_1/MatMulMatMuldropout_2/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	β
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
τ
c
*__inference_dropout_2_layer_call_fn_502048

inputs
identity’StatefulPartitionedCallΓ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_500830o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
ώ

Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_500707

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Π
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Πl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Πb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Π~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Πw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Πv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Πw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Πg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Π
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
ό

O__inference_layer_normalization_layer_call_and_return_conditional_losses_500628

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Π
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Πl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Π*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Πb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Π~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Πw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Πv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Πw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Πg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Π
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
2
ω
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_501842	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Π
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Π
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ΠJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5?d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Π
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΠΠ*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΠΠZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΠΠ^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:¦
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΠΠ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=Θ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΠΠ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΠΠ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΠΠ¦
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Π*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Π*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΠΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π:?????????Π: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Π

_user_specified_namequery:SO
,
_output_shapes
:?????????Π

_user_specified_namevalue
²*
ω
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_500581	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Π
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Π
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Π*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ΠJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5?d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Π
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΠΠ*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΠΠs
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΠΠ¦
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Π*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Π*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Πl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΠΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π:?????????Π: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Π

_user_specified_namequery:SO
,
_output_shapes
:?????????Π

_user_specified_namevalue
>


A__inference_model_layer_call_and_return_conditional_losses_501120

inputs1
multi_head_attention_501066:-
multi_head_attention_501068:1
multi_head_attention_501070:-
multi_head_attention_501072:1
multi_head_attention_501074:-
multi_head_attention_501076:1
multi_head_attention_501078:)
multi_head_attention_501080:(
layer_normalization_501084:(
layer_normalization_501086:#
conv1d_501090:
conv1d_501092:%
conv1d_1_501096:
conv1d_1_501098:*
layer_normalization_1_501101:*
layer_normalization_1_501103:
dense_501108:	Π

dense_501110:
 
dense_1_501114:
	
dense_1_501116:	
identity’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dropout/StatefulPartitionedCall’!dropout_1/StatefulPartitionedCall’!dropout_2/StatefulPartitionedCall’+layer_normalization/StatefulPartitionedCall’-layer_normalization_1/StatefulPartitionedCall’,multi_head_attention/StatefulPartitionedCallλ
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinputsinputsmulti_head_attention_501066multi_head_attention_501068multi_head_attention_501070multi_head_attention_501072multi_head_attention_501074multi_head_attention_501076multi_head_attention_501078multi_head_attention_501080*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_500997ύ
dropout/StatefulPartitionedCallStatefulPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_500926Ζ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0layer_normalization_501084layer_normalization_501086*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_500628
tf.__operators__.add/AddV2AddV24layer_normalization/StatefulPartitionedCall:output:0inputs*
T0*,
_output_shapes
:?????????Π
conv1d/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_501090conv1d_501092*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_500651
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_500883
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_1_501096conv1d_1_501098*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_500679Ο
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0layer_normalization_1_501101layer_normalization_1_501103*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_500707΄
tf.__operators__.add_1/AddV2AddV26layer_normalization_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Πφ
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_500536
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_501108dense_501110*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_500726
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_500830
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_1_501114dense_1_501116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_500750w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	Γ
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
ο
 
&__inference_model_layer_call_fn_500800
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:	Π


unknown_16:


unknown_17:
	

unknown_18:	
identity’StatefulPartitionedCallΝ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_500757o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????Π
!
_user_specified_name	input_1
Ξ

B__inference_conv1d_layer_call_and_return_conditional_losses_500651

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Π
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Π*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Π*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΠU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????Πf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????Π
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
σ	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_502065

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs

κ
5__inference_multi_head_attention_layer_call_fn_501765	
query	
value
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity’StatefulPartitionedCallΒ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_500997t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Π`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π:?????????Π: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????Π

_user_specified_namequery:SO
,
_output_shapes
:?????????Π

_user_specified_namevalue
ζ9
©	
A__inference_model_layer_call_and_return_conditional_losses_501265
input_11
multi_head_attention_501211:-
multi_head_attention_501213:1
multi_head_attention_501215:-
multi_head_attention_501217:1
multi_head_attention_501219:-
multi_head_attention_501221:1
multi_head_attention_501223:)
multi_head_attention_501225:(
layer_normalization_501229:(
layer_normalization_501231:#
conv1d_501235:
conv1d_501237:%
conv1d_1_501241:
conv1d_1_501243:*
layer_normalization_1_501246:*
layer_normalization_1_501248:
dense_501253:	Π

dense_501255:
 
dense_1_501259:
	
dense_1_501261:	
identity’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’+layer_normalization/StatefulPartitionedCall’-layer_normalization_1/StatefulPartitionedCall’,multi_head_attention/StatefulPartitionedCallν
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinput_1input_1multi_head_attention_501211multi_head_attention_501213multi_head_attention_501215multi_head_attention_501217multi_head_attention_501219multi_head_attention_501221multi_head_attention_501223multi_head_attention_501225*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_500581ν
dropout/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_500604Ύ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0layer_normalization_501229layer_normalization_501231*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_500628
tf.__operators__.add/AddV2AddV24layer_normalization/StatefulPartitionedCall:output:0input_1*
T0*,
_output_shapes
:?????????Π
conv1d/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_501235conv1d_501237*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_500651γ
dropout_1/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_500662
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_1_501241conv1d_1_501243*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_500679Ο
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0layer_normalization_1_501246layer_normalization_1_501248*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_500707΄
tf.__operators__.add_1/AddV2AddV26layer_normalization_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Πφ
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_500536
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_501253dense_501255*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_500726έ
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_500737
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_1_501259dense_1_501261*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_500750w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	Ω
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall:U Q
,
_output_shapes
:?????????Π
!
_user_specified_name	input_1
σ	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_500830

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
Ά
F
*__inference_dropout_1_layer_call_fn_501930

inputs
identityΈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_500662e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????Π"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
Ξ

B__inference_conv1d_layer_call_and_return_conditional_losses_501925

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Π
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Π*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Π*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΠU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????Πf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????Π
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs

p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_500536

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
λ

4__inference_layer_normalization_layer_call_fn_501878

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallμ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Π*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_500628t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Π`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Π: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
μ
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_501940

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Π`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Π"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
μ

&__inference_model_layer_call_fn_501373

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:	Π


unknown_16:


unknown_17:
	

unknown_18:	
identity’StatefulPartitionedCallΜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_500757o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Π: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Π
 
_user_specified_nameinputs
Γ

(__inference_dense_1_layer_call_fn_502074

inputs
unknown:
	
	unknown_0:	
identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_500750o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs


σ
A__inference_dense_layer_call_and_return_conditional_losses_500726

inputs1
matmul_readvariableop_resource:	Π
-
biasadd_readvariableop_resource:

identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Π
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Π: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????Π
 
_user_specified_nameinputs
Ψ
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_502053

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????
[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs

ύ.
"__inference__traced_restore_502532
file_prefix8
*assignvariableop_layer_normalization_gamma:9
+assignvariableop_1_layer_normalization_beta:6
 assignvariableop_2_conv1d_kernel:,
assignvariableop_3_conv1d_bias:8
"assignvariableop_4_conv1d_1_kernel:.
 assignvariableop_5_conv1d_1_bias:<
.assignvariableop_6_layer_normalization_1_gamma:;
-assignvariableop_7_layer_normalization_1_beta:2
assignvariableop_8_dense_kernel:	Π
+
assignvariableop_9_dense_bias:
4
"assignvariableop_10_dense_1_kernel:
	.
 assignvariableop_11_dense_1_bias:	'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: K
5assignvariableop_17_multi_head_attention_query_kernel:E
3assignvariableop_18_multi_head_attention_query_bias:I
3assignvariableop_19_multi_head_attention_key_kernel:C
1assignvariableop_20_multi_head_attention_key_bias:K
5assignvariableop_21_multi_head_attention_value_kernel:E
3assignvariableop_22_multi_head_attention_value_bias:V
@assignvariableop_23_multi_head_attention_attention_output_kernel:L
>assignvariableop_24_multi_head_attention_attention_output_bias:#
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: B
4assignvariableop_29_adam_layer_normalization_gamma_m:A
3assignvariableop_30_adam_layer_normalization_beta_m:>
(assignvariableop_31_adam_conv1d_kernel_m:4
&assignvariableop_32_adam_conv1d_bias_m:@
*assignvariableop_33_adam_conv1d_1_kernel_m:6
(assignvariableop_34_adam_conv1d_1_bias_m:D
6assignvariableop_35_adam_layer_normalization_1_gamma_m:C
5assignvariableop_36_adam_layer_normalization_1_beta_m::
'assignvariableop_37_adam_dense_kernel_m:	Π
3
%assignvariableop_38_adam_dense_bias_m:
;
)assignvariableop_39_adam_dense_1_kernel_m:
	5
'assignvariableop_40_adam_dense_1_bias_m:	R
<assignvariableop_41_adam_multi_head_attention_query_kernel_m:L
:assignvariableop_42_adam_multi_head_attention_query_bias_m:P
:assignvariableop_43_adam_multi_head_attention_key_kernel_m:J
8assignvariableop_44_adam_multi_head_attention_key_bias_m:R
<assignvariableop_45_adam_multi_head_attention_value_kernel_m:L
:assignvariableop_46_adam_multi_head_attention_value_bias_m:]
Gassignvariableop_47_adam_multi_head_attention_attention_output_kernel_m:S
Eassignvariableop_48_adam_multi_head_attention_attention_output_bias_m:B
4assignvariableop_49_adam_layer_normalization_gamma_v:A
3assignvariableop_50_adam_layer_normalization_beta_v:>
(assignvariableop_51_adam_conv1d_kernel_v:4
&assignvariableop_52_adam_conv1d_bias_v:@
*assignvariableop_53_adam_conv1d_1_kernel_v:6
(assignvariableop_54_adam_conv1d_1_bias_v:D
6assignvariableop_55_adam_layer_normalization_1_gamma_v:C
5assignvariableop_56_adam_layer_normalization_1_beta_v::
'assignvariableop_57_adam_dense_kernel_v:	Π
3
%assignvariableop_58_adam_dense_bias_v:
;
)assignvariableop_59_adam_dense_1_kernel_v:
	5
'assignvariableop_60_adam_dense_1_bias_v:	R
<assignvariableop_61_adam_multi_head_attention_query_kernel_v:L
:assignvariableop_62_adam_multi_head_attention_query_bias_v:P
:assignvariableop_63_adam_multi_head_attention_key_kernel_v:J
8assignvariableop_64_adam_multi_head_attention_key_bias_v:R
<assignvariableop_65_adam_multi_head_attention_value_kernel_v:L
:assignvariableop_66_adam_multi_head_attention_value_bias_v:]
Gassignvariableop_67_adam_multi_head_attention_attention_output_kernel_v:S
Eassignvariableop_68_adam_multi_head_attention_attention_output_bias_v:
identity_70’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_64’AssignVariableOp_65’AssignVariableOp_66’AssignVariableOp_67’AssignVariableOp_68’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9 $
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*Ζ#
valueΌ#BΉ#FB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*‘
valueBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*T
dtypesJ
H2F	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv1d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv1d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp.assignvariableop_6_layer_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp-assignvariableop_7_layer_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_17AssignVariableOp5assignvariableop_17_multi_head_attention_query_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_18AssignVariableOp3assignvariableop_18_multi_head_attention_query_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_19AssignVariableOp3assignvariableop_19_multi_head_attention_key_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:’
AssignVariableOp_20AssignVariableOp1assignvariableop_20_multi_head_attention_key_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_21AssignVariableOp5assignvariableop_21_multi_head_attention_value_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_22AssignVariableOp3assignvariableop_22_multi_head_attention_value_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_23AssignVariableOp@assignvariableop_23_multi_head_attention_attention_output_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_24AssignVariableOp>assignvariableop_24_multi_head_attention_attention_output_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_layer_normalization_gamma_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_30AssignVariableOp3assignvariableop_30_adam_layer_normalization_beta_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_conv1d_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_conv1d_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv1d_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_layer_normalization_1_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_layer_normalization_1_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_dense_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_41AssignVariableOp<assignvariableop_41_adam_multi_head_attention_query_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_42AssignVariableOp:assignvariableop_42_adam_multi_head_attention_query_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_43AssignVariableOp:assignvariableop_43_adam_multi_head_attention_key_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_44AssignVariableOp8assignvariableop_44_adam_multi_head_attention_key_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_45AssignVariableOp<assignvariableop_45_adam_multi_head_attention_value_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_46AssignVariableOp:assignvariableop_46_adam_multi_head_attention_value_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Έ
AssignVariableOp_47AssignVariableOpGassignvariableop_47_adam_multi_head_attention_attention_output_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ά
AssignVariableOp_48AssignVariableOpEassignvariableop_48_adam_multi_head_attention_attention_output_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_layer_normalization_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_50AssignVariableOp3assignvariableop_50_adam_layer_normalization_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_conv1d_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_conv1d_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_conv1d_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv1d_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_layer_normalization_1_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_layer_normalization_1_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adam_dense_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp%assignvariableop_58_adam_dense_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_1_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_1_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_61AssignVariableOp<assignvariableop_61_adam_multi_head_attention_query_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_62AssignVariableOp:assignvariableop_62_adam_multi_head_attention_query_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_63AssignVariableOp:assignvariableop_63_adam_multi_head_attention_key_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_64AssignVariableOp8assignvariableop_64_adam_multi_head_attention_key_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_65AssignVariableOp<assignvariableop_65_adam_multi_head_attention_value_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_66AssignVariableOp:assignvariableop_66_adam_multi_head_attention_value_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Έ
AssignVariableOp_67AssignVariableOpGassignvariableop_67_adam_multi_head_attention_attention_output_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ά
AssignVariableOp_68AssignVariableOpEassignvariableop_68_adam_multi_head_attention_attention_output_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ½
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_70IdentityIdentity_69:output:0^NoOp_1*
T0*
_output_shapes
: ͺ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_70Identity_70:output:0*‘
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ψ
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_500737

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????
[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*―
serving_default
@
input_15
serving_default_input_1:0?????????Π;
dense_10
StatefulPartitionedCall:0?????????	tensorflow/serving/predict:Α
Σ
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer

_query_dense

_key_dense
_value_dense
_softmax
_dropout_layer
_output_dense
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
Δ
*axis
	+gamma
,beta
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
(
3	keras_api"
_tf_keras_layer
»

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
Δ
Jaxis
	Kgamma
Lbeta
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
(
S	keras_api"
_tf_keras_layer
₯
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
»

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
γ
piter

qbeta_1

rbeta_2
	sdecay
tlearning_rate+m,m4m5mBmCmKmLmZm[mhmimumvmwmxmymzm {m‘|m’+v£,v€4v₯5v¦Bv§Cv¨Kv©LvͺZv«[v¬hv­iv?uv―vv°wv±xv²yv³zv΄{v΅|vΆ"
	optimizer
Ά
u0
v1
w2
x3
y4
z5
{6
|7
+8
,9
410
511
B12
C13
K14
L15
Z16
[17
h18
i19"
trackable_list_wrapper
Ά
u0
v1
w2
x3
y4
z5
{6
|7
+8
,9
410
511
B12
C13
K14
L15
Z16
[17
h18
i19"
trackable_list_wrapper
 "
trackable_list_wrapper
Μ
}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ζ2γ
&__inference_model_layer_call_fn_500800
&__inference_model_layer_call_fn_501373
&__inference_model_layer_call_fn_501418
&__inference_model_layer_call_fn_501208ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
?2Ο
A__inference_model_layer_call_and_return_conditional_losses_501532
A__inference_model_layer_call_and_return_conditional_losses_501674
A__inference_model_layer_call_and_return_conditional_losses_501265
A__inference_model_layer_call_and_return_conditional_losses_501322ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΜBΙ
!__inference__wrapped_model_500526input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
-
serving_default"
signature_map
τ
partial_output_shape
full_output_shape

ukernel
vbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
τ
partial_output_shape
full_output_shape

wkernel
xbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
τ
partial_output_shape
full_output_shape

ykernel
zbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
«
‘	variables
’trainable_variables
£regularization_losses
€	keras_api
₯__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
τ
§partial_output_shape
¨full_output_shape

{kernel
|bias
©	variables
ͺtrainable_variables
«regularization_losses
¬	keras_api
­__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
X
u0
v1
w2
x3
y4
z5
{6
|7"
trackable_list_wrapper
X
u0
v1
w2
x3
y4
z5
{6
|7"
trackable_list_wrapper
 "
trackable_list_wrapper
²
―non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
π2ν
5__inference_multi_head_attention_layer_call_fn_501743
5__inference_multi_head_attention_layer_call_fn_501765ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¦2£
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_501800
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_501842ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
΄non_trainable_variables
΅layers
Άmetrics
 ·layer_regularization_losses
Έlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
2
(__inference_dropout_layer_call_fn_501847
(__inference_dropout_layer_call_fn_501852΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Δ2Α
C__inference_dropout_layer_call_and_return_conditional_losses_501857
C__inference_dropout_layer_call_and_return_conditional_losses_501869΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
':%2layer_normalization/gamma
&:$2layer_normalization/beta
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ήnon_trainable_variables
Ίlayers
»metrics
 Όlayer_regularization_losses
½layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
ή2Ϋ
4__inference_layer_normalization_layer_call_fn_501878’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ω2φ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_501900’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
"
_generic_user_object
#:!2conv1d/kernel
:2conv1d/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ύnon_trainable_variables
Ώlayers
ΐmetrics
 Αlayer_regularization_losses
Βlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ρ2Ξ
'__inference_conv1d_layer_call_fn_501909’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
μ2ι
B__inference_conv1d_layer_call_and_return_conditional_losses_501925’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Γnon_trainable_variables
Δlayers
Εmetrics
 Ζlayer_regularization_losses
Ηlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
2
*__inference_dropout_1_layer_call_fn_501930
*__inference_dropout_1_layer_call_fn_501935΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Θ2Ε
E__inference_dropout_1_layer_call_and_return_conditional_losses_501940
E__inference_dropout_1_layer_call_and_return_conditional_losses_501952΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
%:#2conv1d_1/kernel
:2conv1d_1/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Θnon_trainable_variables
Ιlayers
Κmetrics
 Λlayer_regularization_losses
Μlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
Σ2Π
)__inference_conv1d_1_layer_call_fn_501961’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv1d_1_layer_call_and_return_conditional_losses_501976’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
):'2layer_normalization_1/gamma
(:&2layer_normalization_1/beta
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Νnon_trainable_variables
Ξlayers
Οmetrics
 Πlayer_regularization_losses
Ρlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
ΰ2έ
6__inference_layer_normalization_1_layer_call_fn_501985’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϋ2ψ
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_502007’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
?non_trainable_variables
Σlayers
Τmetrics
 Υlayer_regularization_losses
Φlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
π2ν
9__inference_global_average_pooling1d_layer_call_fn_502012―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_502018―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
:	Π
2dense/kernel
:
2
dense/bias
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Χnon_trainable_variables
Ψlayers
Ωmetrics
 Ϊlayer_regularization_losses
Ϋlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
Π2Ν
&__inference_dense_layer_call_fn_502027’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
λ2θ
A__inference_dense_layer_call_and_return_conditional_losses_502038’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
άnon_trainable_variables
έlayers
ήmetrics
 ίlayer_regularization_losses
ΰlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
2
*__inference_dropout_2_layer_call_fn_502043
*__inference_dropout_2_layer_call_fn_502048΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Θ2Ε
E__inference_dropout_2_layer_call_and_return_conditional_losses_502053
E__inference_dropout_2_layer_call_and_return_conditional_losses_502065΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 :
	2dense_1/kernel
:	2dense_1/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
αnon_trainable_variables
βlayers
γmetrics
 δlayer_regularization_losses
εlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
?2Ο
(__inference_dense_1_layer_call_fn_502074’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ν2κ
C__inference_dense_1_layer_call_and_return_conditional_losses_502085’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
7:52!multi_head_attention/query/kernel
1:/2multi_head_attention/query/bias
5:32multi_head_attention/key/kernel
/:-2multi_head_attention/key/bias
7:52!multi_head_attention/value/kernel
1:/2multi_head_attention/value/bias
B:@2,multi_head_attention/attention_output/kernel
8:62*multi_head_attention/attention_output/bias
 "
trackable_list_wrapper

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
13"
trackable_list_wrapper
0
ζ0
η1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΛBΘ
$__inference_signature_wrapper_501721input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
θnon_trainable_variables
ιlayers
κmetrics
 λlayer_regularization_losses
μlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
νnon_trainable_variables
ξlayers
οmetrics
 πlayer_regularization_losses
ρlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ςnon_trainable_variables
σlayers
τmetrics
 υlayer_regularization_losses
φlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
χnon_trainable_variables
ψlayers
ωmetrics
 ϊlayer_regularization_losses
ϋlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
όnon_trainable_variables
ύlayers
ώmetrics
 ?layer_regularization_losses
layer_metrics
‘	variables
’trainable_variables
£regularization_losses
₯__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
©	variables
ͺtrainable_variables
«regularization_losses
­__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
R

total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
,:*2 Adam/layer_normalization/gamma/m
+:)2Adam/layer_normalization/beta/m
(:&2Adam/conv1d/kernel/m
:2Adam/conv1d/bias/m
*:(2Adam/conv1d_1/kernel/m
 :2Adam/conv1d_1/bias/m
.:,2"Adam/layer_normalization_1/gamma/m
-:+2!Adam/layer_normalization_1/beta/m
$:"	Π
2Adam/dense/kernel/m
:
2Adam/dense/bias/m
%:#
	2Adam/dense_1/kernel/m
:	2Adam/dense_1/bias/m
<::2(Adam/multi_head_attention/query/kernel/m
6:42&Adam/multi_head_attention/query/bias/m
::82&Adam/multi_head_attention/key/kernel/m
4:22$Adam/multi_head_attention/key/bias/m
<::2(Adam/multi_head_attention/value/kernel/m
6:42&Adam/multi_head_attention/value/bias/m
G:E23Adam/multi_head_attention/attention_output/kernel/m
=:;21Adam/multi_head_attention/attention_output/bias/m
,:*2 Adam/layer_normalization/gamma/v
+:)2Adam/layer_normalization/beta/v
(:&2Adam/conv1d/kernel/v
:2Adam/conv1d/bias/v
*:(2Adam/conv1d_1/kernel/v
 :2Adam/conv1d_1/bias/v
.:,2"Adam/layer_normalization_1/gamma/v
-:+2!Adam/layer_normalization_1/beta/v
$:"	Π
2Adam/dense/kernel/v
:
2Adam/dense/bias/v
%:#
	2Adam/dense_1/kernel/v
:	2Adam/dense_1/bias/v
<::2(Adam/multi_head_attention/query/kernel/v
6:42&Adam/multi_head_attention/query/bias/v
::82&Adam/multi_head_attention/key/kernel/v
4:22$Adam/multi_head_attention/key/bias/v
<::2(Adam/multi_head_attention/value/kernel/v
6:42&Adam/multi_head_attention/value/bias/v
G:E23Adam/multi_head_attention/attention_output/kernel/v
=:;21Adam/multi_head_attention/attention_output/bias/v¦
!__inference__wrapped_model_500526uvwxyz{|+,45BCKLZ[hi5’2
+’(
&#
input_1?????????Π
ͺ "1ͺ.
,
dense_1!
dense_1?????????	?
D__inference_conv1d_1_layer_call_and_return_conditional_losses_501976fBC4’1
*’'
%"
inputs?????????Π
ͺ "*’'
 
0?????????Π
 
)__inference_conv1d_1_layer_call_fn_501961YBC4’1
*’'
%"
inputs?????????Π
ͺ "?????????Π¬
B__inference_conv1d_layer_call_and_return_conditional_losses_501925f454’1
*’'
%"
inputs?????????Π
ͺ "*’'
 
0?????????Π
 
'__inference_conv1d_layer_call_fn_501909Y454’1
*’'
%"
inputs?????????Π
ͺ "?????????Π£
C__inference_dense_1_layer_call_and_return_conditional_losses_502085\hi/’,
%’"
 
inputs?????????

ͺ "%’"

0?????????	
 {
(__inference_dense_1_layer_call_fn_502074Ohi/’,
%’"
 
inputs?????????

ͺ "?????????	’
A__inference_dense_layer_call_and_return_conditional_losses_502038]Z[0’-
&’#
!
inputs?????????Π
ͺ "%’"

0?????????

 z
&__inference_dense_layer_call_fn_502027PZ[0’-
&’#
!
inputs?????????Π
ͺ "?????????
―
E__inference_dropout_1_layer_call_and_return_conditional_losses_501940f8’5
.’+
%"
inputs?????????Π
p 
ͺ "*’'
 
0?????????Π
 ―
E__inference_dropout_1_layer_call_and_return_conditional_losses_501952f8’5
.’+
%"
inputs?????????Π
p
ͺ "*’'
 
0?????????Π
 
*__inference_dropout_1_layer_call_fn_501930Y8’5
.’+
%"
inputs?????????Π
p 
ͺ "?????????Π
*__inference_dropout_1_layer_call_fn_501935Y8’5
.’+
%"
inputs?????????Π
p
ͺ "?????????Π₯
E__inference_dropout_2_layer_call_and_return_conditional_losses_502053\3’0
)’&
 
inputs?????????

p 
ͺ "%’"

0?????????

 ₯
E__inference_dropout_2_layer_call_and_return_conditional_losses_502065\3’0
)’&
 
inputs?????????

p
ͺ "%’"

0?????????

 }
*__inference_dropout_2_layer_call_fn_502043O3’0
)’&
 
inputs?????????

p 
ͺ "?????????
}
*__inference_dropout_2_layer_call_fn_502048O3’0
)’&
 
inputs?????????

p
ͺ "?????????
­
C__inference_dropout_layer_call_and_return_conditional_losses_501857f8’5
.’+
%"
inputs?????????Π
p 
ͺ "*’'
 
0?????????Π
 ­
C__inference_dropout_layer_call_and_return_conditional_losses_501869f8’5
.’+
%"
inputs?????????Π
p
ͺ "*’'
 
0?????????Π
 
(__inference_dropout_layer_call_fn_501847Y8’5
.’+
%"
inputs?????????Π
p 
ͺ "?????????Π
(__inference_dropout_layer_call_fn_501852Y8’5
.’+
%"
inputs?????????Π
p
ͺ "?????????ΠΣ
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_502018{I’F
?’<
63
inputs'???????????????????????????

 
ͺ ".’+
$!
0??????????????????
 «
9__inference_global_average_pooling1d_layer_call_fn_502012nI’F
?’<
63
inputs'???????????????????????????

 
ͺ "!??????????????????»
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_502007fKL4’1
*’'
%"
inputs?????????Π
ͺ "*’'
 
0?????????Π
 
6__inference_layer_normalization_1_layer_call_fn_501985YKL4’1
*’'
%"
inputs?????????Π
ͺ "?????????ΠΉ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_501900f+,4’1
*’'
%"
inputs?????????Π
ͺ "*’'
 
0?????????Π
 
4__inference_layer_normalization_layer_call_fn_501878Y+,4’1
*’'
%"
inputs?????????Π
ͺ "?????????ΠΑ
A__inference_model_layer_call_and_return_conditional_losses_501265|uvwxyz{|+,45BCKLZ[hi=’:
3’0
&#
input_1?????????Π
p 

 
ͺ "%’"

0?????????	
 Α
A__inference_model_layer_call_and_return_conditional_losses_501322|uvwxyz{|+,45BCKLZ[hi=’:
3’0
&#
input_1?????????Π
p

 
ͺ "%’"

0?????????	
 ΐ
A__inference_model_layer_call_and_return_conditional_losses_501532{uvwxyz{|+,45BCKLZ[hi<’9
2’/
%"
inputs?????????Π
p 

 
ͺ "%’"

0?????????	
 ΐ
A__inference_model_layer_call_and_return_conditional_losses_501674{uvwxyz{|+,45BCKLZ[hi<’9
2’/
%"
inputs?????????Π
p

 
ͺ "%’"

0?????????	
 
&__inference_model_layer_call_fn_500800ouvwxyz{|+,45BCKLZ[hi=’:
3’0
&#
input_1?????????Π
p 

 
ͺ "?????????	
&__inference_model_layer_call_fn_501208ouvwxyz{|+,45BCKLZ[hi=’:
3’0
&#
input_1?????????Π
p

 
ͺ "?????????	
&__inference_model_layer_call_fn_501373nuvwxyz{|+,45BCKLZ[hi<’9
2’/
%"
inputs?????????Π
p 

 
ͺ "?????????	
&__inference_model_layer_call_fn_501418nuvwxyz{|+,45BCKLZ[hi<’9
2’/
%"
inputs?????????Π
p

 
ͺ "?????????	φ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_501800‘uvwxyz{|i’f
_’\
$!
query?????????Π
$!
value?????????Π

 

 
p 
p 
ͺ "*’'
 
0?????????Π
 φ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_501842‘uvwxyz{|i’f
_’\
$!
query?????????Π
$!
value?????????Π

 

 
p 
p
ͺ "*’'
 
0?????????Π
 Ξ
5__inference_multi_head_attention_layer_call_fn_501743uvwxyz{|i’f
_’\
$!
query?????????Π
$!
value?????????Π

 

 
p 
p 
ͺ "?????????ΠΞ
5__inference_multi_head_attention_layer_call_fn_501765uvwxyz{|i’f
_’\
$!
query?????????Π
$!
value?????????Π

 

 
p 
p
ͺ "?????????Π΄
$__inference_signature_wrapper_501721uvwxyz{|+,45BCKLZ[hi@’=
’ 
6ͺ3
1
input_1&#
input_1?????????Π"1ͺ.
,
dense_1!
dense_1?????????	