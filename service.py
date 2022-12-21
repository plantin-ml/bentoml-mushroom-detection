import glob
import io
import os

import bentoml
import requests
import torch
from bentoml.io import JSON, Image
from PIL import Image as PilImage
from pydantic import BaseModel, HttpUrl


def get_latest_model(search_dir='.'):
    last_list = glob.glob(f'{search_dir}/*best.pt', recursive=True)

    return max(last_list, key=os.path.getctime) if last_list else ''


class Yolov5Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        model_path = get_latest_model('./weights')

        self.model = torch.hub.load(
            repo_or_dir="./yolov5",
            source='local',
            model='custom',
            path=model_path,
        )

        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

        # Config inference settings
        self.inference_size = 640
        self.model.model.warmup(imgsz=(1, 3, self.inference_size, self.inference_size))  # warmup

        # Optional configs
        # self.model.conf = 0.25  # NMS confidence threshold
        # self.model.iou = 0.45  # NMS IoU threshold
        # self.model.agnostic = False  # NMS class-agnostic
        # self.model.multi_label = False  # NMS multiple labels per box
        # self.model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        # self.model.max_det = 1000  # maximum number of detections per image
        # self.model.amp = False  # Automatic Mixed Precision (AMP) inference

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, input_imgs):
        # Return predictions only
        outputs = self.model(input_imgs, size=self.inference_size)

        results = []
        for det in outputs.pred:
            detections = []
            for i in det:
                d = {}
                d['name'] = outputs.names[int(i[5])]

                d['xmin'] = float(i[0])
                d['ymin'] = float(i[1])
                d['xmax'] = float(i[2])
                d['ymax'] = float(i[3])

                d['confidence'] = i[4].tolist()
                detections.append(d)
            results.append(detections)

        return results

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def render(self, input_imgs):
        # Return images with boxes and labels
        return self.model(input_imgs, size=self.inference_size).render()


yolo_v5_runner = bentoml.Runner(Yolov5Runnable, max_batch_size=30, name='yolo_runner')

svc = bentoml.Service("bento_mushroom_detection", runners=[yolo_v5_runner])


class ImageFeatures(BaseModel):
    image_url: HttpUrl

@svc.api(input=JSON(pydantic_model=ImageFeatures), output=JSON())
async def detect_by_url(input_data):
    r = requests.get(input_data.image_url, stream=True)
    r.raise_for_status()

    with io.BytesIO(r.content) as f:
        input_img = PilImage.open(f).convert('RGB')

    batch_ret = await yolo_v5_runner.inference.async_run([input_img])
    return batch_ret[0]

@svc.api(input=Image(), output=JSON())
async def detect_by_file(input_img):
    batch_ret = await yolo_v5_runner.inference.async_run([input_img])
    return batch_ret[0]

@svc.api(input=Image(), output=Image())
async def render(input_img):
    batch_ret = await yolo_v5_runner.render.async_run([input_img])
    return batch_ret[0]