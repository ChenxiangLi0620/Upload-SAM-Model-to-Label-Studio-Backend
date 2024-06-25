# Upload-SAM-Model-to-Label-Studio-Backend

### 1.
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
### 2.
```bash
cd playground/label_anything
nano app.py
```
Create app.py with following code:
```python
import argparse
import json
import logging
import logging.config
import os
import sys
from flask import Flask
from label_studio_ml.api import init_app

# Ensure the correct path for custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam'))
from mmdetection import MMDetection

# Logging configuration
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'stream': 'ext://sys.stdout',
            'formatter': 'standard'
        }
    },
    'root': {
        'level': 'ERROR',
        'handlers': ['console'],
        'propagate': True
    }
})

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return dict()
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config

# Argument parsing
parser = argparse.ArgumentParser(description='Label studio')
parser.add_argument('-p', '--port', dest='port', type=int, default=9090, help='Server port')
parser.add_argument('--host', dest='host', type=str, default='0.0.0.0', help='Server host')
parser.add_argument('--kwargs', '--with', dest='kwargs', metavar='KEY=VAL', nargs='+',
                    type=lambda kv: kv.split('='), help='Additional LabelStudioMLBase model initialization kwargs')
parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Switch debug mode')
parser.add_argument('--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    default=None, help='Logging level')
parser.add_argument('--model-dir', dest='model_dir', default=os.path.dirname(__file__), help='Directory models are store')
parser.add_argument('--check', dest='check', action='store_true', help='Validate model instance before launching server')

args = parser.parse_args()

# Setup logging level
if args.log_level:
    logging.root.setLevel(args.log_level)

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_kwargs():
    param = dict()
    for k, v in args.kwargs:
        if v.isdigit():
            param[k] = int(v)
        elif v.lower() == 'true':
            param[k] = True
        elif v.lower() == 'false':
            param[k] = False
        elif isfloat(v):
            param[k] = float(v)
        else:
            param[k] = v
    return param

kwargs = get_kwargs_from_config()

if args.kwargs:
    kwargs.update(parse_kwargs())

if args.check:
    print(f'Check "{MMDetection.__name__}" instance creation..')
    model = MMDetection(**kwargs)

app = init_app(
    model_class=MMDetection,
    model_dir=os.environ.get('MODEL_DIR', args.model_dir),
    redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
    redis_host=os.environ.get('REDIS_HOST', 'localhost'),
    redis_port=os.environ.get('REDIS_PORT', 6379),
    **kwargs)

if __name__ == '__main__':
    app.run(host=args.host, port=args.port, debug=args.debug)
else:
    app = init_app(
        model_class=MMDetection,
        model_dir=os.environ.get('MODEL_DIR', os.path.dirname(__file__)),
        redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=os.environ.get('REDIS_PORT', 6379))

```
