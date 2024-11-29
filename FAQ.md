## ModuleNotFoundError: No module named 'matcha'

Matcha-TTS is a third_party module. Please check `third_party` directory. If there is no `Matcha-TTS`, execute `git submodule update --init --recursive`.

run `export PYTHONPATH=third_party/Matcha-TTS` if you want to use `from cosyvoice.cli.cosyvoice import CosyVoice` in python script.

## cannot find resource.zip or cannot unzip resource.zip

Please make sure you have git-lfs installed. Execute

```sh
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl
```

## CUDA_HOME does not exist, unable to compile CUDA op(s)

查看conda环境中的cuda版本
conda list | grep cuda
nvcc --version
which nvcc
如果 nvcc 报错，没有安装编译器
通过以下命令安装：

```bash
conda install -c nvidia cuda-compiler
```

再次执行
nvcc --version

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_12_02:18:05_PDT_2024
Cuda compilation tools, release 12.6, V12.6.77
Build cuda_12.6.r12.6/compiler.34841621_0
```

查看路径：
which nvcc