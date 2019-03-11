import os


if __name__ == "__main__":
    os.system(
        'python3  -m torch.distributed.launch  --nnodes=2  --node_rank=0 --nproc_per_node=2  --master_addr="10.63.3.65" --master_port="12345"  '
        '/home/zhai/PycharmProjects/Demo35/pytorch_Faster_tool/train_M_GPU_M_node/Mask_Rcnn.py ')

