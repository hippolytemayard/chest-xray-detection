import torch

from chest_xray_detection.ml_detection_api.configs.settings import logging
from chest_xray_detection.ml_detection_api.domain.wrappers.base_wrapper import BaseModelWrapper
from chest_xray_detection.ml_detection_develop.dataset.transforms.utils import (
    instantiate_transforms_from_config,
)
from chest_xray_detection.ml_detection_develop.models.faster_rcnn import get_faster_rcnn
from chest_xray_detection.ml_detection_api.utils.objects.prediction import ObjectDetectionFormat
from chest_xray_detection.ml_detection_api.utils.objects.base_objects import BBoxPrediction

from chest_xray_detection.ml_detection_api.utils.formatting import convert_to_api_format


class MultiClassDetectionWrapper(BaseModelWrapper):
    @classmethod
    def load(cls, config):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # if config.FROM_S3:
        #    logging.info(f"Loading model {config.CKPT} from S3..")
        #    saved_model_dict = load_model_from_s3(
        #        bucket_name=AWS_S3_MODEL_BUCKET_NAME, filepath=config.CKPT
        #    )
        # else:

        saved_model_dict = torch.load(config.CKPT, map_location=device)

        logging.info("Initializing faster_rcnn model and loading weights")

        model = get_faster_rcnn(
            backbone=config.BACKBONE,
            pretrained=False,
            num_classes=config.NUM_CLASSES,
        ).to(device)

        model.load_state_dict(saved_model_dict["model"])
        logging.info("Weights loaded!")

        return cls(
            model=model,
            device=device,
            config=config,
        )

    def before_inference(self, image):

        transforms = instantiate_transforms_from_config(self.config.TRANSFORMS.INFERENCE)
        print(transforms)
        image = transforms(image)
        print(type(image))
        image = image.unsqueeze(0) / 255.0
        image = image.to(self.device)
        return image

    def inference(self, processed_input: torch.Tensor) -> list[ObjectDetectionFormat]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(processed_input)

        print(outputs)
        return [ObjectDetectionFormat(**output) for output in outputs]

    def after_inference(self, outputs: list[ObjectDetectionFormat]) -> ObjectDetectionFormat:

        logging.info("Postprocessing predictions..")
        output = outputs[0]
        output.to_cpu()
        output.filter_by_proba(config=self.config.POSTPROCESSING)
        output.nms_on_boxes(iou_threshold=self.config.POSTPROCESSING.NMS_IOU_THRESHOLDS)
        # print(output)
        print(f"Postprocessing output ==> {output}")
        return output

    def convert_output(self, output: ObjectDetectionFormat) -> list[BBoxPrediction]:

        logging.info("Formatting predictions predictions..")
        print(f"convert_output ==> {output}")
        list_detections = convert_to_api_format(
            output=output,
            classes_list=self.config.CLASSES,
            model_name=self.config.NAME,
        )
        # list_missing_tooth = add_keys(list_missing_tooth)

        if len(list_detections) == 0:
            logging.info("No anomaly detected!")
        return list_detections

    def __call__(self, image) -> list[BBoxPrediction]:
        processed_input = self.before_inference(image=image)
        outputs = self.inference(processed_input=processed_input)
        processed_output = self.after_inference(outputs=outputs)
        converted_output = self.convert_output(processed_output)
        return converted_output
