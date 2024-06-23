import torch

from chest_xray_detection.ml_detection_api.configs.settings import logging
from chest_xray_detection.ml_detection_api.domain.wrappers.base_wrapper import BaseModelWrapper
from chest_xray_detection.ml_detection_api.utils.formatting import convert_to_api_format
from chest_xray_detection.ml_detection_api.utils.objects.base_objects import BBoxPrediction
from chest_xray_detection.ml_detection_api.utils.objects.prediction import ObjectDetectionFormat
from chest_xray_detection.ml_detection_develop.dataset.transforms.utils import (
    instantiate_transforms_from_config,
)
from chest_xray_detection.ml_detection_develop.models.detection.faster_rcnn import get_faster_rcnn


class MultiClassDetectionWrapper(BaseModelWrapper):
    """
    Wrapper class for multi-class object detection models using Faster R-CNN.

    This class handles model loading, inference, and post-processing of predictions.

    Attributes:
        model (torch.nn.Module): The object detection model.
        device (torch.device): The device (CPU or GPU) on which the model runs.
        config: Configuration settings for the model.
        debug (bool): Flag indicating whether to enable debug mode.
    """

    @classmethod
    def load(cls, config, debug: bool = False) -> "MultiClassDetectionWrapper":
        """
        Load the model and initialize the wrapper.

        Args:
            config: Configuration object containing model parameters.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.

        Returns:
            MultiClassDetectionWrapper: Initialized wrapper instance.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved_model_dict = torch.load(config.CKPT, map_location=device)

        logging.info("Initializing faster_rcnn model and loading weights")
        model = get_faster_rcnn(
            backbone=config.BACKBONE,
            pretrained=False,
            num_classes=config.NUM_CLASSES,
        ).to(device)

        model.load_state_dict(saved_model_dict["model"])
        logging.info("Weights loaded!")

        return cls(model=model, device=device, config=config, debug=debug)

    def before_inference(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input image before inference.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Processed image tensor ready for inference.
        """
        transforms = instantiate_transforms_from_config(
            self.config.TRANSFORMS.INFERENCE, task=self.config.TRANSFORMS.TASK
        )
        image = transforms(image)
        image = image.unsqueeze(0) / 255.0
        image = image.to(self.device)
        return image

    def inference(self, processed_input: torch.Tensor) -> list[ObjectDetectionFormat]:
        """
        Perform inference on the processed input tensor.

        Args:
            processed_input (torch.Tensor): Processed input tensor.

        Returns:
            list[ObjectDetectionFormat]: List of ObjectDetectionFormat objects containing model predictions.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(processed_input)

        return [ObjectDetectionFormat(**output) for output in outputs]

    def after_inference(self, outputs: list[ObjectDetectionFormat]) -> ObjectDetectionFormat:
        """
        Post-process the model predictions.

        Args:
            outputs (list[ObjectDetectionFormat]): List of prediction objects.

        Returns:
            ObjectDetectionFormat: Post-processed prediction object.
        """
        logging.info("Postprocessing predictions..")
        output = outputs[0]
        output.to_cpu()
        output.filter_by_proba(config=self.config.POSTPROCESSING)
        output.nms_on_boxes(iou_threshold=self.config.POSTPROCESSING.NMS_IOU_THRESHOLDS)

        return output

    def convert_output(self, output: ObjectDetectionFormat) -> list[BBoxPrediction]:
        """
        Convert model predictions to API-compatible format.

        Args:
            output (ObjectDetectionFormat): Post-processed prediction object.

        Returns:
            list[BBoxPrediction]: List of BBoxPrediction objects.
        """
        logging.info("Formatting predictions..")
        list_detections = convert_to_api_format(
            output=output,
            classes_list=self.config.CLASSES,
            model_name=self.config.NAME,
        )
        if len(list_detections) == 0:
            logging.info("No anomaly detected!")
        return list_detections

    def __call__(self, image: torch.Tensor) -> list[BBoxPrediction]:
        """
        Perform inference and convert predictions in a single call.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            list[BBoxPrediction]: List of BBoxPrediction objects representing detected anomalies.
        """
        processed_input = self.before_inference(image=image)
        print(processed_input.shape)
        outputs = self.inference(processed_input=processed_input)
        processed_output = self.after_inference(outputs=outputs)
        converted_output = self.convert_output(processed_output)
        if self.debug:
            logging.info(f"output : {outputs}")
            logging.info(f"processed_output : {processed_output}")
            logging.info(f"converted_output : {converted_output}")
        return converted_output
