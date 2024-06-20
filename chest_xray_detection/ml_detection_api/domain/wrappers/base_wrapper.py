from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig

from chest_xray_detection.ml_detection_api.utils.objects.prediction import ObjectDetectionFormat


class BaseModelWrapper(ABC):
    """Abstract Base Class (ABC) for wrapping model.

    This object wraps any model in order to perform an inference from A to Z.

    It also contains its own ThreadPoolExecutor for handling multithreading scenario.

    3 principal methods  :
        - before_inference : takes for input a np.darray and performs any preprocessing needed on
            this array
        - inference : performs the actual inference, takes for input the output of before_inference
        - after_inference : takes for input the output of inference and postprocess the output
            to refine the output usually with tensor/array-based operation
        - convert_output : takes as input the output of after_inference and convert it to a different
            final format used by end user/another piece of code

    2 principal methods for multithreading concurrent model __call__:
        - threaded_call : execute __call__ in an internal thread and return a future
        - shutdown : close ThreadPoolExecutor (important if server shutdown)
    """

    def __init__(
        self,
        model: Callable,
        config: DictConfig,
        device: torch.device,
        metadata: Any | None = None,
        max_workers: int = 1,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.metadata = metadata

        # thread pool executor
        thread_name_prefix = type(self).__name__.replace("Wrapper", "Thread")
        self.executor = ThreadPoolExecutor(max_workers, thread_name_prefix=thread_name_prefix)

        self.debug = debug

    def threaded_call(self, *args, **kwargs) -> Future:
        """Submit __call__ in the thread pool executor for concurrent calls. Pass same arguments as __call__."""
        return self.executor.submit(self.__call__, *args, **kwargs)

    def shutdown(self) -> None:
        """Shutdown threadpool executor. WARNING -> very important for service graceful shutdown."""
        self.executor.shutdown()

    @classmethod
    def load(cls, config: DictConfig, postprocessing: bool = True):
        """Class method that instanciate a wrapper with a specific config"""
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self, inputs: list[np.ndarray | torch.Tensor]
    ) -> list[list[ObjectDetectionFormat]]:
        """Chains the wrapper methods.

        Parameters
        ----------
        inputs : list[np.ndarray | torch.Tensor]
            Input of the wrapper

        Returns
        -------
        list[list[BaseObjectType]]
            Postprocessed output of the network
        """
        raise NotImplementedError()


class ObjectDetectionWrapper(BaseModelWrapper):
    def __init__(self, model, device, config, metadata, postprocessing: bool = True):
        super().__init__(model=model, device=device, config=config, metadata=metadata)
        self.postprocessing = postprocessing

    @abstractmethod
    def before_inference(self, inputs: list[np.ndarray] | list[torch.Tensor]) -> list[torch.Tensor]:
        """Define the operation to perform on the input before inference

        Parameters
        ----------
        inputs : list[np.ndarray] | list[torch.Tensor]
            Input of the wrapper

        Returns
        -------
        list[torch.Tensor]
            Input of the wrapper processed and ready to be fed to the model
        """
        raise NotImplementedError

    @abstractmethod
    def inference(
        self,
        processed_inputs: list[torch.Tensor],
    ) -> list[ObjectDetectionFormat]:
        """Perform the actual inference

        Parameters
        ----------
        processed_inputs : list[torch.Tensor]
            Processed_inputs coming from before_inference

        Returns
        -------
        list[InstanceSegmentationFormat]
            Result of the network inference
        """
        raise NotImplementedError

    @abstractmethod
    def after_inference(
        self,
        inference_outputs: list[ObjectDetectionFormat],
    ) -> list[ObjectDetectionFormat]:
        """Postprocess the output of the network. No formatting logic.

        Parameters
        ----------
        inference_outputs : list[InstanceSegmentationFormat]
            Output of the network to be postprocessed

        Returns
        -------
        list[InstanceSegmentationFormat]
            Postprocessed output of the network
        """
        raise NotImplementedError

    @abstractmethod
    def convert_output(
        self,
        after_inference_outputs: list[ObjectDetectionFormat],
    ) -> list[list[ObjectDetectionFormat]]:
        """Convert the output of the network into a different format.

        Can exceptionnaly include some post-processing based on converted format but
        ideally, all post-processing should be located in "after_inference" if possible.

        Parameters
        ----------
        after_inference_outputs : list[InstanceSegmentationFormat]
            Output of the network (potentially postprocessed)

        Returns
        -------
        list[list[ObjectDetectionFormat]]
            Converted and formatted output of the model wrapper
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self, inputs: list[np.ndarray | torch.Tensor]
    ) -> list[list[ObjectDetectionFormat]]:
        """Chains the wrapper methods.

        Parameters
        ----------
        inputs : list[np.ndarray | torch.Tensor]
            Input of the wrapper

        Returns
        -------
        list[list[BaseObjectType]]
            Postprocessed output of the network
        """
        processed_inputs = self.before_inference(inputs=inputs)
        inference_outputs = self.inference(processed_inputs=processed_inputs, image_sizes=[])
        processed_output = self.after_inference(inference_outputs=inference_outputs, image_sizes=[])
        converted_output = self.convert_output(
            after_inference_outputs=processed_output, roi_boxes=[]
        )
        return converted_output
