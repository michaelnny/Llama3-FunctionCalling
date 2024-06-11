# Install DeepSpeed on OpenSuse

Install Python dev packages

```bash
sudo zypper install python3-devel libaio-devel

```

Install CUDA Toolkit, follow the guide:

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

```bash

# Install CUDA toolkit

wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda-repo-opensuse15-12-5-local-12.5.0_555.42.02-1.x86_64.rpmsudo

rpm -i cuda-repo-opensuse15-12-5-local-12.5.0_555.42.02-1.x86_64.rpm

sudo zypper refresh

sudo zypper install cuda-toolkit-12-1


# Install CUDA driver

sudo zypper install cuda-drivers

```


Then, we can install the DeepSpeed package
```bash

pip3 install deepspeed

```
