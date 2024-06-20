from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from torchvision.ops import nms


@dataclass
class ObjectDetectionFormat:
    """
    The standard format for Object Detection in the pipeline.

    This format follows the Faster R-CNN prediction format.

    Attributes:
        boxes (torch.Tensor): A tensor of shape [N, 4] containing the coordinates of detected boxes.
        scores (torch.Tensor): A tensor of shape [N] containing the scores associated with each box.
        labels (torch.Tensor): A tensor of shape [N] containing the labels associated with each box.
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

    @property
    def is_empty(self) -> bool:
        return len(self.boxes) == 0

    @property
    def device(self) -> torch.device:
        return self.boxes.device

    def filter_indexes(self, indexes: torch.Tensor) -> torch.Tensor:
        """Filters out in place the prediction by specified indexes
        Can be used for a reordering or making a subset

        Parameters
        ----------
        indexes : torch.Tensor
            List of indexes to perform the operation with
        """
        self.boxes = self.boxes[indexes]
        self.scores = self.scores[indexes]
        self.labels = self.labels[indexes]
        return indexes

    def filter_by_proba(self, config: DictConfig) -> torch.Tensor:
        """Filter object by probability"""
        if self.is_empty:
            return torch.tensor([])
        to_keep_indexes = np.stack(
            [
                (
                    score
                    >= (
                        config.THRESHOLD
                        if config.get("THRESHOLD")
                        else config.CLASSES[int(label.item())].PROBA_THRESHOLD
                    )
                ).cpu()
                for score, label in zip(self.scores, self.labels)
            ]
        )
        print(config.THRESHOLD)
        return self.filter_indexes(torch.tensor(to_keep_indexes))

    def nms_on_boxes(self, iou_threshold: float = 0.5) -> Optional[torch.Tensor]:
        if self.is_empty:
            return
        nms_indexes = nms(boxes=self.boxes, scores=self.scores, iou_threshold=iou_threshold)
        self.filter_indexes(nms_indexes)

    def ignore_indexes(self, classes: DictConfig) -> torch.Tensor:
        """Ignore specified labels

        Parameters
        ----------
        classes : DictConfig
            The classes

        Returns
        -------
        torch.Tensor
            The indexes to keep
        """
        ignored_labels = []
        for label, classe in classes.items():
            if classe.get("IGNORED"):
                ignored_labels.append(label)

        indexes_to_keep = torch.ones(len(self.labels), dtype=torch.bool).to(self.labels.device)
        for label in ignored_labels:
            indexes_to_keep *= ~torch.eq(self.labels, label)

        return self.filter_indexes(indexes=indexes_to_keep)

    def to_cpu(self):
        """Move predictions to cpu"""
        self.boxes = self.boxes.to(torch.device("cpu"))
        self.scores = self.scores.to(torch.device("cpu"))
        self.labels = self.labels.to(torch.device("cpu"))

    def to_tensor(self) -> dict[str, torch.Tensor]:
        """Convert predictions to tensors"""
        output = {
            "boxes": self.boxes,
            "labels": self.labels,
            "scores": self.scores,
        }
        return output

    def threshold_scores(self, config: DictConfig):
        """Ensure that scores are always above minimum_score"""
        self.scores = torch.maximum(self.scores, torch.tensor(config.MINIMUM_SCORE))
