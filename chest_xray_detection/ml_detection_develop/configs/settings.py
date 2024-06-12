from pathlib import Path
from chest_xray_detection.ml_detection_develop.utils.files import load_yaml

DATA_PATH = Path("/home/ubuntu/data/images")
ANNOTATION_PATH = Path("/home/ubuntu/data/BBox_List_2017.csv")

LABEL_MAPPING_DICT = load_yaml(path=Path(__file__).parent / "label_mapping.yaml")
