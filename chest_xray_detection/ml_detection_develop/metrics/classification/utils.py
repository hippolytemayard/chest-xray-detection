import logging
from typing import Union

import torchmetrics
import torchmetrics.classification
import torchmetrics.detection
from omegaconf import OmegaConf
from torchmetrics import MetricCollection


def instantiate_metrics_from_config(
    metrics_config: dict[str, list[str | dict]]
) -> MetricCollection:
    """
    Instantiate metrics from a configuration dictionary.

    Args:
        metrics_config (dict[str, list[Union[str, dict]]]): A dictionary
            containing information about metrics. It should have keys:
            'frameworks', 'classes', and 'classes_params'. 'frameworks'
            should contain names of frameworks (e.g., 'torchmetrics'),
            'classes' should contain names of metric classes, and
            'classes_params' should contain dictionaries with parameters
            for metric classes.

    Returns:
        MetricCollection: A collection of instantiated metrics.

    Raises:
        ValueError: If an error occurs during the metric instantiation process.
    """
    compositions = []
    try:
        for framework, fn_name, fn_params in zip(
            metrics_config["frameworks"],
            metrics_config["classes"],
            metrics_config["classes_params"],
        ):
            fn_params = OmegaConf.to_object(fn_params)
            if framework == "torchmetrics":
                fn = getattr(torchmetrics.classification, fn_name)(**fn_params)

            compositions.append(fn)

    except Exception as e:
        logging.info(f"Got error on {fn_name} with kwargs: {fn_params}")
        raise ValueError(f"Exception: {e}")

    logging.info(f"composition metrics: {compositions}")

    return MetricCollection(compositions)
