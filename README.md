# Upload-SAM-Model-to-Label-Studio-Backend

###1.
```bash
conda create -n rtmdet-sam python=3.9 -y
conda activate rtmdet-sam
conda install git
git clone https://github.com/open-mmlab/playground

# Linux and Windows CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

cd playground/label_anything
# Before proceeding to the next step in Windows, you need to complete the following command line.
# conda install pycocotools -c conda-forge
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install segment-anything-hq

# download HQ-SAM pretrained model
#wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth
#wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth

pip install label-studio==1.7.3
pip install label-studio-ml==1.0.9
```
