# Upload-SAM-Model-to-Label-Studio-Backend

### 1. Environment Configuration
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
### 2. ML Backend and Model Loading Setup
```bash
cd playground/label_anything
nano app.py
```
Create app.py with following code:
```python
# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os
import argparse
import json
import logging
import random
import string
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
import torch
from botocore.exceptions import ClientError
from label_studio_converter import brush
from label_studio_ml.api import init_app
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (DATA_UNDEFINED_NAME, get_image_size,
                                   get_single_tag_keys)
from label_studio_tools.core.utils.io import get_data_dir

# Add the directory containing mmdetection.py to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam'))

from mmdetection import MMDetection  # Now you can import it correctly
from filter_poly import NearNeighborRemover  # Ensure the path to filter_poly.py is correct

# Use the standard logging.config
import logging.config

logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'standard': {
            'format':
            '[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s'  # noqa E501
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label studio')
    parser.add_argument('-p', '--port', dest='port', type=int, default=9090, help='Server port')
    parser.add_argument('--host', dest='host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--kwargs', '--with', dest='kwargs', metavar='KEY=VAL', nargs='+',
                        type=lambda kv: kv.split('='), help='Additional LabelStudioMLBase model initialization kwargs')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Switch debug mode')
    parser.add_argument('--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=None,
                        help='Logging level')
    parser.add_argument('--model-dir', dest='model_dir', default=os.path.dirname(__file__),
                        help='Directory models are store')
    parser.add_argument('--check', dest='check', action='store_true', help='Validate model instance before launching server')

    args = parser.parse_args()

    # setup logging level
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
            elif v == 'True' or v == 'true':
                param[k] = True
            elif v == 'False' or v == 'False':
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
        print('Check "' + MMDetection.__name__ + '" instance creation..')
        model = MMDetection(**kwargs)

    app = init_app(
        model_class=MMDetection,
        model_dir=os.environ.get('MODEL_DIR', args.model_dir),
        redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=os.environ.get('REDIS_PORT', 6379),
        **kwargs)

    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    # for uWSGI use
    app = init_app(
        model_class=MMDetection,
        model_dir=os.environ.get('MODEL_DIR', os.path.dirname(__file__)),
        redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=os.environ.get('REDIS_PORT', 6379))

```
Then, go to
```bash
cd sam
nano mmdetection.py
```
and update the model configuration
```python
# Copyright (c) OpenMMLab. All rights reserved.
import io
import json
import logging
import os
import random
import string
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
import torch
from botocore.exceptions import ClientError
from label_studio_converter import brush
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (DATA_UNDEFINED_NAME, get_image_size,
                                   get_single_tag_keys)
from label_studio_tools.core.utils.io import get_data_dir
from filter_poly import NearNeighborRemover
import pdb
# from mmdet.apis import inference_detector, init_detector

logger = logging.getLogger(__name__)


def load_my_model(
        model_name="sam_hq",
        device="cuda:0",
        sam_config="vit_h",
        sam_checkpoint_file="/home/hammond/playground/label_anything/sam_hq_vit_h.pth"):
    """
    Loads the Segment Anything model on initializing Label studio, so if you call it outside MyModel it doesn't load every time you try to make a prediction
    Returns the predictor object. For more, look at Facebook's SAM docs
    """
    if model_name == "sam":
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except:
            raise ModuleNotFoundError(
                "segment_anything is not installed, run `pip install segment_anything` to install")
        sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint_file)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    elif model_name == "sam_hq":
        from segment_anything_hq import sam_model_registry, SamPredictor
        sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint_file)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    elif model_name == "mobile_sam":
        from models.mobile_sam import SamPredictor, sam_model_registry
        sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint_file)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    else:
        raise NotImplementedError


class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection."""

    def __init__(self,
                 model_name="sam_hq",
                 config_file=None,
                 checkpoint_file=None,
                 sam_config='vit_h',
                 sam_checkpoint_file=None,
                 image_dir=None,
                 labels_file=None,
                 out_mask=True,
                 out_bbox=False,
                 out_poly=False,
                 score_threshold=0.5,
                 device='cpu',
                 **kwargs):

        super(MMDetection, self).__init__(**kwargs)

        PREDICTOR = load_my_model(
            model_name, device, sam_config, sam_checkpoint_file)
        self.PREDICTOR = PREDICTOR

        self.out_mask = out_mask
        self.out_bbox = out_bbox
        self.out_poly = out_poly
        self.labels_file = labels_file
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(
            f'{self.__class__.__name__} reads images from {self.image_dir}')
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.labels_in_config = dict(
            label=self.parsed_label_config.get('KeyPointLabels', [])
        )

        if 'RectangleLabels' in self.parsed_label_config and self.out_bbox:

            self.parsed_label_config_RectangleLabels = {
                'RectangleLabels': self.parsed_label_config['RectangleLabels']
            }
            self.from_name_RectangleLabels, self.to_name_RectangleLabels, self.value_RectangleLabels, self.labels_in_config_RectangleLabels = get_single_tag_keys(  # noqa E501
                self.parsed_label_config_RectangleLabels, 'RectangleLabels', 'Image')

        if 'BrushLabels' in self.parsed_label_config:

            self.parsed_label_config_BrushLabels = {
                'BrushLabels': self.parsed_label_config['BrushLabels']
            }
            self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(  # noqa E501
                self.parsed_label_config_BrushLabels, 'BrushLabels', 'Image')

        if 'BrushLabels' in self.parsed_label_config:

            self.parsed_label_config_BrushLabels = {
                'BrushLabels': self.parsed_label_config['BrushLabels']
            }
            self.from_name_BrushLabels, self.to_name_BrushLabels, self.value_BrushLabels, self.labels_in_config_BrushLabels = get_single_tag_keys(  # noqa E501
                self.parsed_label_config_BrushLabels, 'BrushLabels', 'Image')

        if 'PolygonLabels' in self.parsed_label_config and self.out_poly:

            self.parsed_label_config_PolygonLabels = {
                'PolygonLabels': self.parsed_label_config['PolygonLabels']
            }
            self.from_name_PolygonLabels, self.to_name_PolygonLabels, self.value_PolygonLabels, self.labels_in_config_PolygonLabels = get_single_tag_keys(  # noqa E501
                self.parsed_label_config_PolygonLabels, 'PolygonLabels', 'Image')

        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag # noqa E501
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values',
                                                       '').split(','):
                    self.label_map[predicted_value] = label_name

        self.score_thresh = score_threshold

    def _get_image_url(self, task):
        image_url = task['data'].get(
            self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={
                        'Bucket': bucket_name,
                        'Key': key
                    })
            except ClientError as exc:
                logger.warning(
                    f'Can\'t generate presigned URL for {image_url}. Reason: {exc}'  # noqa E501
                )
        return image_url

    def predict(self, tasks, **kwargs):

        predictor = self.PREDICTOR

        results = []
        assert len(tasks) == 1
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)

        if kwargs.get('context') is None:
            return []

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        prompt_type = kwargs['context']['result'][0]['type']
        original_height = kwargs['context']['result'][0]['original_height']
        original_width = kwargs['context']['result'][0]['original_width']

        if prompt_type == 'keypointlabels':
            # getting x and y coordinates of the keypoint
            x = kwargs['context']['result'][0]['value']['x'] * \
                original_width / 100
            y = kwargs['context']['result'][0]['value']['y'] * \
                original_height / 100
            output_label = kwargs['context']['result'][0]['value']['labels'][0]

            masks, scores, logits = predictor.predict(
                point_coords=np.array([[x, y]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )

        if prompt_type == 'rectanglelabels':

            x = kwargs['context']['result'][0]['value']['x'] * \
                original_width / 100
            y = kwargs['context']['result'][0]['value']['y'] * \
                original_height / 100
            w = kwargs['context']['result'][0]['value']['width'] * \
                original_width / 100
            h = kwargs['context']['result'][0]['value']['height'] * \
                original_height / 100

            output_label = kwargs['context']['result'][0]['value']['rectanglelabels'][0]

            masks, scores, logits = predictor.predict(
                box=np.array([x, y, x+w, y+h]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
        mask = masks[0].astype(np.uint8)  # each mask has shape [H, W]

        # Find contours
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.out_bbox:
            new_contours = []
            for contour in contours:
                new_contours.extend(list(contour))
            new_contours = np.array(new_contours)
            x, y, w, h = cv2.boundingRect(new_contours)
            results.append({
                'from_name': self.from_name_RectangleLabels,
                'to_name': self.to_name_RectangleLabels,
                'type': 'rectanglelabels',
                'value': {
                    'rectanglelabels': [output_label],
                    'x': float(x) / original_width * 100,
                    'y': float(y) / original_height * 100,
                    'width': float(w) / original_width * 100,
                    'height': float(h) / original_height * 100,
                },
                "id": ''.join(random.SystemRandom().choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6)), # creates a random ID for your label every time
            })

        if self.out_poly:

            points_list = []
            for contour in contours:
                points = []
                for point in contour:
                    x, y = point[0]
                    points.append([float(x)/original_width*100,
                                  float(y)/original_height * 100])
                points_list.extend(points)
            filterd_points=NearNeighborRemover(distance_threshold=0.4).remove_near_neighbors(points_list) # remove near neighbors (increase distance_threshold to reduce more points)
            results.append({
                "from_name": self.from_name_PolygonLabels,
                "to_name": self.to_name_PolygonLabels,
                "original_width": original_width,
                "original_height": original_height,
                "value": {
                    "points": filterd_points,
                    "polygonlabels": [output_label],
                },
                "type": "polygonlabels",
                "id": ''.join(random.SystemRandom().choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6)), # creates a random ID for your label every time
                "readonly": False,
            })

        if self.out_mask:
            mask = mask * 255
            rle = brush.mask2rle(mask)

            results.append({
                "from_name": self.from_name_BrushLabels,
                "to_name": self.to_name_BrushLabels,
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": [output_label],
                },
                "type": "brushlabels",
                "id": ''.join(random.SystemRandom().choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6)), # creates a random ID for your label every time
                "readonly": False,
            })

        return [{'result': results}]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
```
Start the backend by
```bash
cd ..
pip install timm
python app.py --port 8003 --host 0.0.0.0 --with sam_config=vit_h sam_checkpoint_file=./sam_hq_vit_h.pth out_mask=True out_bbox=True device=cuda:0 model_name=sam_hq
```
### 3. Setup in Label Anything
Open a new terminal, connect to the server, activate the environment again, and then run
```bash
cd playground/label_anything
# Linux requires the following commands
export ML_TIMEOUT_SETUP=40
# Windows requires the following commands
set ML_TIMEOUT_SETUP=40
label-studio start
```
In label studio, create a new project, upload dataset, and then custom Labeling Interface by modifying and adapting the following code according to the dataset:
```XML
<View>
  <Image name="image" value="$image" zoom="true"/>
  <KeyPointLabels name="KeyPointLabels" toName="image">
    <Label value="cat" smart="true" background="#e51515" showInline="true"/>
    <Label value="person" smart="true" background="#412cdd" showInline="true"/>
  </KeyPointLabels>
  <RectangleLabels name="RectangleLabels" toName="image">
  	<Label value="cat" background="#FF0000"/>
  	<Label value="person" background="#0d14d3"/>
  </RectangleLabels>
  <PolygonLabels name="PolygonLabels" toName="image">
  	<Label value="cat" background="#FF0000"/>
  	<Label value="person" background="#0d14d3"/>
  </PolygonLabels>
  <BrushLabels name="BrushLabels" toName="image">
  	<Label value="cat" background="#FF0000"/>
  	<Label value="person" background="#0d14d3"/>
  </BrushLabels>
</View>
```
Once the dataset is uploaded and labeling interface is set up in Settings, add the model in Machine Learning with URL: 
http://localhost:8003 and hit Save.

![image](https://github.com/ChenxiangLi0620/Upload-SAM-Model-to-Label-Studio-Backend/assets/168608052/8062ac44-dda1-49d0-b219-c9a9f4ad475c)

When one starts labeling, toggle Auto-Annotation and Auto Accept Annotation Suggestions on, select one of the tools (keypoints/bounding box/brush) and their relative label to annotate the image.

![image](https://github.com/ChenxiangLi0620/Upload-SAM-Model-to-Label-Studio-Backend/assets/168608052/95c5778f-b5d9-4ac5-a584-9a879d07ac6d)

### Acknowledgments

This project makes use of code from the following repositories:

- [OpenMMLab PlayGround: Semi-Automated Annotation with Label-Studio and SAM]([https://github.com/username/repository](https://github.com/open-mmlab/playground/blob/main/label_anything/readme.md]) - Adapted part of the code about environment configuration including depnedencies installation and pretrained model downloading.
