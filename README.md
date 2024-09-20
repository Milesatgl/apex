# 版本说明
## 适配环境：
win10、pytorch2.0.0+cu118、Python3.10.10

## 菜鸟免责声明
全是GPT帮我的，俺也不知道对不对

## Debug 记录：
1、一开始出现的是`'['ninja', '-v']' returned non-zero exit status 1`，太误导了，一开始以为是 ninja 的问题，后来才发现不是。这个命令来自于 `\torch\utils\cpp_extension.py` 中的 `_run_ninja_build`函数，这个函数只执行`ninja -v`命令，具体的编译细节是由`apex\build\temp.win-amd64-cpython-310\Release\build.ninja`的`build.ninja`来决定的，所以手动运行`ninja -v`会出现缺少`build.ninja`的提示。这个误导的原因，我感觉可能是多核处理的时候，真正的编译错误被快速覆盖过去了，在 terminal 中来不及看到就没了，而且也没办法把所有的编译信息保存下来。

2、针对问题 1，需要把`\torch\utils\cpp_extension.py`，`class BuildExtension(build_ext)`的`init`实现中`self.use_ninja = kwargs.get('use_ninja', True)`的 True 改为 False，即默认关闭ninja编译功能，这样就默认一个文件一个文件的编译，便于定位错误。

3、后续就是一个错误一个错误的排查了

3.1
`error: static assertion failed with "You've instantiated std::aligned_storage<Len, Align> with an extended alignment (in other words, Align > alignof(max_align_t)). Before VS 2017 15.8, the member "type" would non-conformingly have an alignment of only alignof(max_align_t). VS 2017 15.8 was fixed to handle this correctly, but the fix inherently changes layout and breaks binary compatibility (*only* for uses of aligned_storage with extended alignments). Please define either (1) _ENABLE_EXTENDED_ALIGNED_STORAGE to acknowledge that you understand this message and that you actually want a type with an extended alignment, or (2) _DISABLE_EXTENDED_ALIGNED_STORAGE to silence this message and get the old non-conforming behavior."`

这个错误来自编译器遇到 std::aligned_storage 时的一项特性行为变化。大概意思就是，在 VS 2017 版本 15.8 之前，std::aligned_storage 对于大 alignof(max_align_t) 的对齐要求，非标准地采用了 alignof(max_align_t)。但是自 VS 2017 版本 15.8 起，修复了这一行为，使得对齐要求得到正确处理，但这意味着在某些情况下可能会影响二进制兼容性。

从错误信息中，VS 提供了两个选项来处理这个问题：

- 定义 _ENABLE_EXTENDED_ALIGNED_STORAGE，表示你理解并希望使用如此处理大对齐要求的类型。
- 定义 _DISABLE_EXTENDED_ALIGNED_STORAGE，以保持旧的行为，这可能导致与标准不一致的对齐，但维持旧的二进制兼容性。

解决方案是在`setup.py`文件中，相关文件编译命令的`extra_compile_args`中添加`["-D_ENABLE_EXTENDED_ALIGNED_STORAGE"]`

3.2 剩下就是单个文件的debug了，代码已编译测试通过。(ps：setup.py 中的 cuda_lib_dir 是我自己电脑的路径，需要按需修改)

# Introduction

This repository holds NVIDIA-maintained utilities to streamline mixed precision and distributed training in Pytorch.
Some of the code here will be included in upstream Pytorch eventually.
The intent of Apex is to make up-to-date utilities available to users as quickly as possible.

## Full API Documentation: [https://nvidia.github.io/apex](https://nvidia.github.io/apex)

## [GTC 2019](https://github.com/mcarilli/mixed_precision_references/tree/master/GTC_2019) and [Pytorch DevCon 2019](https://github.com/mcarilli/mixed_precision_references/tree/master/Pytorch_Devcon_2019) Slides

# Contents

## 1. Amp:  Automatic Mixed Precision

**Deprecated. Use [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)**

`apex.amp` is a tool to enable mixed precision training by changing only 3 lines of your script.
Users can easily experiment with different pure and mixed precision training modes by supplying
different flags to `amp.initialize`.

[Webinar introducing Amp](https://info.nvidia.com/webinar-mixed-precision-with-pytorch-reg-page.html)
(The flag `cast_batchnorm` has been renamed to `keep_batchnorm_fp32`).

[API Documentation](https://nvidia.github.io/apex/amp.html)

[Comprehensive Imagenet example](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

[DCGAN example coming soon...](https://github.com/NVIDIA/apex/tree/master/examples/dcgan)

[Moving to the new Amp API](https://nvidia.github.io/apex/amp.html#transition-guide-for-old-api-users) (for users of the deprecated "Amp" and "FP16_Optimizer" APIs)

## 2. Distributed Training

**`apex.parallel.DistributedDataParallel` is deprecated. Use [`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel)**

`apex.parallel.DistributedDataParallel` is a module wrapper, similar to
`torch.nn.parallel.DistributedDataParallel`.  It enables convenient multiprocess distributed training,
optimized for NVIDIA's NCCL communication library.

[API Documentation](https://nvidia.github.io/apex/parallel.html)

[Python Source](https://github.com/NVIDIA/apex/tree/master/apex/parallel)

[Example/Walkthrough](https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed)

The [Imagenet example](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)
shows use of `apex.parallel.DistributedDataParallel` along with `apex.amp`.

### Synchronized Batch Normalization

**Deprecated. Use [`torch.nn.SyncBatchNorm`](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html)**

`apex.parallel.SyncBatchNorm` extends `torch.nn.modules.batchnorm._BatchNorm` to
support synchronized BN.
It allreduces stats across processes during multiprocess (DistributedDataParallel) training.
Synchronous BN has been used in cases where only a small
local minibatch can fit on each GPU.
Allreduced stats increase the effective batch size for the BN layer to the
global batch size across all processes (which, technically, is the correct
formulation).
Synchronous BN has been observed to improve converged accuracy in some of our research models.

### Checkpointing

To properly save and load your `amp` training, we introduce the `amp.state_dict()`, which contains all `loss_scalers` and their corresponding unskipped steps,
as well as `amp.load_state_dict()` to restore these attributes.

In order to get bitwise accuracy, we recommend the following workflow:
```python
# Initialization
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

# Train your model
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
...

# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'amp': amp.state_dict()
}
torch.save(checkpoint, 'amp_checkpoint.pt')
...

# Restore
model = ...
optimizer = ...
checkpoint = torch.load('amp_checkpoint.pt')

model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
amp.load_state_dict(checkpoint['amp'])

# Continue training
...
```

Note that we recommend restoring the model using the same `opt_level`. Also note that we recommend calling the `load_state_dict` methods after `amp.initialize`.

# Installation
Each [`apex.contrib`](./apex/contrib) module requires one or more install options other than `--cpp_ext` and `--cuda_ext`.
Note that contrib modules do not necessarily support stable PyTorch releases.

## Containers
NVIDIA PyTorch Containers are available on NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch.
The containers come with all the custom extensions available at the moment. 

See [the NGC documentation](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) for details such as:
- how to pull a container
- how to run a pulled container
- release notes

## From Source

To install Apex from source, we recommend using the nightly Pytorch obtainable from https://github.com/pytorch/pytorch.

The latest stable release obtainable from https://pytorch.org should also work.

We recommend installing [`Ninja`](https://ninja-build.org/) to make compilation faster.

### Linux
For performance and full functionality, we recommend installing Apex with
CUDA and C++ extensions via
```bash
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

APEX also supports a Python-only build via
```bash
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```
A Python-only build omits:
- Fused kernels required to use `apex.optimizers.FusedAdam`.
- Fused kernels required to use `apex.normalization.FusedLayerNorm` and `apex.normalization.FusedRMSNorm`.
- Fused kernels that improve the performance and numerical stability of `apex.parallel.SyncBatchNorm`.
- Fused kernels that improve the performance of `apex.parallel.DistributedDataParallel` and `apex.amp`.
`DistributedDataParallel`, `amp`, and `SyncBatchNorm` will still be usable, but they may be slower.


### [Experimental] Windows
`pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" .` may work if you were able to build Pytorch from source
on your system. A Python-only build via `pip install -v --no-cache-dir .` is more likely to work.  
If you installed Pytorch in a Conda environment, make sure to install Apex in that same environment.


## Custom C++/CUDA Extensions and Install Options

If a requirement of a module is not met, then it will not be built.

|  Module Name  |  Install Option  |  Misc  |
|---------------|------------------|--------|
|  `apex_C`     |  `--cpp_ext`     | |
|  `amp_C`      |  `--cuda_ext`    | |
|  `syncbn`     |  `--cuda_ext`    | |
|  `fused_layer_norm_cuda`  |  `--cuda_ext`  | [`apex.normalization`](./apex/normalization) |
|  `mlp_cuda`   |  `--cuda_ext`    | |
|  `scaled_upper_triang_masked_softmax_cuda`  |  `--cuda_ext`  | |
|  `generic_scaled_masked_softmax_cuda`  |  `--cuda_ext`  | |
|  `scaled_masked_softmax_cuda`  |  `--cuda_ext`  | |
|  `fused_weight_gradient_mlp_cuda`  |  `--cuda_ext`  | Requires CUDA>=11 |
|  `permutation_search_cuda`  |  `--permutation_search`  | [`apex.contrib.sparsity`](./apex/contrib/sparsity)  |
|  `bnp`        |  `--bnp`         |  [`apex.contrib.groupbn`](./apex/contrib/groupbn) |
|  `xentropy`   |  `--xentropy`    |  [`apex.contrib.xentropy`](./apex/contrib/xentropy)  |
|  `focal_loss_cuda`  |  `--focal_loss`  |  [`apex.contrib.focal_loss`](./apex/contrib/focal_loss)  |
|  `fused_index_mul_2d`  |  `--index_mul_2d`  |  [`apex.contrib.index_mul_2d`](./apex/contrib/index_mul_2d)  |
|  `fused_adam_cuda`  |  `--deprecated_fused_adam`  |  [`apex.contrib.optimizers`](./apex/contrib/optimizers)  |
|  `fused_lamb_cuda`  |  `--deprecated_fused_lamb`  |  [`apex.contrib.optimizers`](./apex/contrib/optimizers)  |
|  `fast_layer_norm`  |  `--fast_layer_norm`  |  [`apex.contrib.layer_norm`](./apex/contrib/layer_norm). different from `fused_layer_norm` |
|  `fmhalib`    |  `--fmha`        |  [`apex.contrib.fmha`](./apex/contrib/fmha)  |
|  `fast_multihead_attn`  |  `--fast_multihead_attn`  |  [`apex.contrib.multihead_attn`](./apex/contrib/multihead_attn)  |
|  `transducer_joint_cuda`  |  `--transducer`  |  [`apex.contrib.transducer`](./apex/contrib/transducer)  |
|  `transducer_loss_cuda`   |  `--transducer`  |  [`apex.contrib.transducer`](./apex/contrib/transducer)  |
|  `cudnn_gbn_lib`  |  `--cudnn_gbn`  | Requires cuDNN>=8.5, [`apex.contrib.cudnn_gbn`](./apex/contrib/cudnn_gbn) |
|  `peer_memory_cuda`  |  `--peer_memory`  |  [`apex.contrib.peer_memory`](./apex/contrib/peer_memory)  |
|  `nccl_p2p_cuda`  |  `--nccl_p2p`  | Requires NCCL >= 2.10, [`apex.contrib.nccl_p2p`](./apex/contrib/nccl_p2p)  |
|  `fast_bottleneck`  |  `--fast_bottleneck`  |  Requires `peer_memory_cuda` and `nccl_p2p_cuda`, [`apex.contrib.bottleneck`](./apex/contrib/bottleneck) |
|  `fused_conv_bias_relu`  |  `--fused_conv_bias_relu`  | Requires cuDNN>=8.4, [`apex.contrib.conv_bias_relu`](./apex/contrib/conv_bias_relu) |
