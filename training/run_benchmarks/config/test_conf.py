'''Test Configs, including'''
# -*-coding:utf-8 -*-

# Set accelerator's vendor name, e.g. iluvatar, cambricon and kunlunxin.
# We will run benchmarks in training/<vendor>
VENDOR = "iluvatar"

# Accelerator options for docker. TODO FIXME support more accelerators.
# possible value of ACCE_CONTAINER_OPT are:
#   iluvatar:
#       ' -v /lib/modules:/lib/modules '
#   kunlunxin:
#       " --device=/dev/xpu0 --device=/dev/xpu1 --device=/dev/xpu2" + \
#       " --device=/dev/xpu3 --device=/dev/xpu4 --device=/dev/xpu5" + \
#       " --device=/dev/xpu6 --device=/dev/xpu7 --device=/dev/xpuctrl"
#   nvidia:
#       " --gpus all"
ACCE_CONTAINER_OPT = " -v /lib/modules:/lib/modules "
# XXX_VISIBLE_DEVICE item name in env
# possible value of ACCE_VISIBLE_DEVICE_ENV_NAME are:
#   CUDA_VISIBLE_DEVICES for nvidia, iluvatar
#   MLU_VISIBLE_DEVICES for cambricon
#   XPU_VISIBLE_DEVICES for kunlunxin
ACCE_VISIBLE_DEVICE_ENV_NAME = "CUDA_VISIBLE_DEVICES"

# Set pip source, which will be used in preparing envs in container
# PIP_SOURCE = "https://mirror.baidu.com/pypi/simple"
# PIP_SOURCE = "https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_SOURCE= "https://mirrors.aliyun.com/pypi/simple/"
# PIP_SOURCE= "http://pypi.douban.com/simple/"



# The path that flagperf deploy in the cluster.
# Users must set FLAGPERF_PATH to where flagperf deploy
# You can assume the preset "/home/FlagPerf/training" points to Null
FLAGPERF_PATH = "/home/yanrui/flagperf/training"
# Set log path on the host here.
FLAGPERF_LOG_PATH = FLAGPERF_PATH + "/result/"

# Set log level. It should be 'debug', 'info', 'warning', or 'error'.
FLAGPERF_LOG_LEVEL = 'debug'

# System config
# Share memory size
SHM_SIZE = "32G"
# Clear cache config. Clean system cache before running testcase.
CLEAR_CACHES = True

# Set the case dict you want to run here.
'''
# Users must use {
    "model:framework:hardwareID:nnodes:nproc:repeat": "dataset path"}
'''
CASES = {
    # "bert:pytorch:A100:1:8:1": "/home/datasets_ckpt/bert/train/",
    # "glm:pytorch:A100:1:8:1": "/home/datasets_ckpt/glm/train/",
    # "cpm:pytorch:A100:1:8:1": "/home/datasets_ckpt/cpm/train/",
    # "resnet50:pytorch:A100:1:8:1": "/home/datasets_ckpt/resnet50/train/",
    "yolov5:pytorch:BI-V100:1:2:1": "/home/yanrui/coco128",
}
