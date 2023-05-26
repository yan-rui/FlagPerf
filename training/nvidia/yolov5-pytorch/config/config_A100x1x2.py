
vendor="nvidia"

data_dir = "/home/datasets_ckpt/yolov5/train" # same as data path in test_conf CASE
n_device=2

epochs= 300
batch_size= 16 * n_device
cfg="yolov5s.yaml"
data="coco.yaml"
hyp="hyps/hyp.scratch-low.yaml"