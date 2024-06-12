import logging
from typing import Union

import torchvision
from omegaconf import OmegaConf
from torchvision.transforms.v2 import Compose
import torchvision.transforms.v2


def instantiate_transforms_from_config(
    transform_config: dict[str, list[Union[str, dict]]]
) -> Compose:
    """
    Instantiate transformations from a configuration dictionary.

    Args:
        transform_config (dict[str, list[Union[str, dict]]]): A dictionary
            containing information about transformations. It should have keys:
            'frameworks', 'classes', and 'classes_params'. 'frameworks' should
            contain names of frameworks (e.g., 'torchvision'), 'classes' should
            contain names of transformation classes, and 'classes_params'
            should contain dictionaries with parameters for transformation
            classes.

    Returns:
        Compose: A composed transformation containing instantiated
            transformation functions.

    Raises:
        ValueError: If an error occurs during the transformation instantiation
            process.
    """
    compositions = []
    try:
        for framework, fn_name, fn_params in zip(
            transform_config["frameworks"],
            transform_config["classes"],
            transform_config["classes_params"],
        ):
            fn_params = OmegaConf.to_object(fn_params)
            if framework == "torchvision":
                fn = getattr(torchvision.transforms.v2, fn_name)(**fn_params)
            compositions.append(fn)

            logging.info(f"compositions: {compositions}")

    except Exception as e:
        logging.info(f"Got error on {fn_name} with kwargs: {fn_params}")
        raise ValueError(f"Exception: {e}")

    logging.info(f"composition transforms: {compositions}")

    return Compose(compositions)
