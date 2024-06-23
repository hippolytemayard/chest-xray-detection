# Training

The object detection dataset is composed of 880 chest X-Rays with 8 different abnormalities.

## Models

Several models have been trained on this task :
- a Faster RCNN with Resnet50-FPN backbone (pretrained on COCO)
- a Faster RCNN with Resnet50-FPN backbone (pretrained on COCO) on the object detection task with merged classes
- a Faster RCNN with MobileNet_v2 backbone
    - MobileNet_v2 backbone has been pretrained on a multi-class multi-label classification task on ChestXray dataset (overlap patients have been excluded)
    - Faster RCNN has been then trained on the object detection task


## Run training

Training has been implemented with a stratified split strategy

Execute the stratified split training script by running the Python file. You can use the command line to specify the path to the configuration file using the --config argument. Here's an example command:

### Faster RCNN with Resnet50-FPN backbone (pretrained on COCO)

```bash
poetry run python chest_xray_detection/ml_detection_develop/train/detection/training.py \
 --config chest_xray_detection/ml_detection_develop/configs/training/detection/training_faster_rcnn.yaml
```

### Single class Faster RCNN with Resnet50-FPN backbone (pretrained on COCO)

```bash
poetry run python chest_xray_detection/ml_detection_develop/train/detection/training.py \
 --config chest_xray_detection/ml_detection_develop/configs/training/detection/training_faster_rcnn_single_class.yaml
```

### Faster RCNN with MobileNet_v2 backbone (classification and then regression)

#### Step 1: train a multi-label multi-class classification model on ChestXray dataset

```bash
poetry run python chest_xray_detection/ml_detection_develop/train/classification/stratified_split_training.py \
 --config chest_xray_detection/ml_detection_develop/configs/training/classification/training_mobilenet_v2.yaml
```

#### Step 2: train an object detection model using pre-trained backbone

```bash
poetry run python chest_xray_detection/ml_detection_develop/train/detection/training.py \
 --config chest_xray_detection/ml_detection_develop/configs/training/detection/training_faster_rcnn_mobilenet.yaml
```

# Evaluation

The evaluation metric that has been chosen for this task is the mean average precision (mAP).
Two different implementation have been compared:
- `Torchmetrics` implementation in `torchmetrics.detection``
- A custom implementation

The Torchmetrics implementation has been used for the benchmark

## Results

| Backbone            | mAP      | mAP@50   | mAP@75   | mAP Medium | mAP Large | mAR@1    | mAR@10   | mAR@100  | mAR Medium | mAR Large |
|---------------------|----------|----------|----------|------------|-----------|----------|----------|----------|------------|-----------|
| **ResNet50-FPN**           | **0.0851**  | **0.2057**   | **0.0615**   | **0.0676**     | **0.0702**    | **0.1768**   | **0.2683**   | **0.2757**   | **0.1736**    | **0.2367**    |
| ResNet50-FPN Single Class | 0.0693   | 0.1781   | 0.0254   | 0.0878     | 0.0710    | 0.0985   | 0.1914   | 0.2406   | 0.2296     | 0.2452    |
| MobileNet (pretrained)**           | 0.0807   | 0.1610   | 0.0711   | 0.0083     | 0.0811    | 0.1545   | 0.2016   | 0.2016   | 0.0281     | 0.2278    |

** The pretrained model evaluation on multi-label multi-class classification

| Model               | MultilabelAUROC      | MultilabelAccuracy@50   |
|---------------------|----------------------|-------------------------|
| ResNet50            | 0.809               | 0.934                     |


## Interpretation

This study results reveal significant variability in the performance of Faster-RCNN architecture with different backbones (even when the task is supposed to be easier, with a single class). Among the evaluated models, ResNet50 backbone emerges as the top performer with **mean average precision (mAP) of 0.0851, mAP@50 score of 0.2057**, and maximum recall (mAR@100) of 0.2757.
The performance may be adversely affected due to the complexity inherent in chest X-rays, which exhibit significant variations in patient anatomy, positioning, and imaging conditions. Moreover, dealing with eight distinct pathology classes introduces additional challenges in terms of annotation diversity and specificity, directly impacting the models' precision and sensitivity.

In order to have a better understanding of the performance of the best model (Faster-RCNN with Resnet50-FPN backbone), we have used the custom implementation of mAP to get the Precision and Recall by class :

| Pathology     | Recall  | Precision | Average Precision (AP) | Total Ground Truth |
|---------------|---------|-----------|------------------------|--------------------|
| Atelectasis   | 0.9231  | 0.0173    | 0.2212                 | 39                 |
| Cardiomegaly  | 1.0     | 0.0919    | 0.5631                 | 26                 |
| Effusion      | 0.9677  | 0.0197    | 0.1816                 | 31                 |
| Infiltrate    | 0.875   | 0.032     | 0.1162                 | 24                 |
| Mass          | 0.75    | 0.037     | 0.2785                 | 20                 |
| Nodule        | 0.7778  | 0.0611    | 0.3231                 | 18                 |
| Pneumonia     | 0.9333  | 0.0223    | 0.2921                 | 15                 |
| Pneumothorax  | 1.0     | 0.0196    | 0.3194                 | 24                 |


The Precision of the model is very poor and the model generates too many False Positives.


## Post-processing

Two different post-processing methods have been implemented at inference time in order to remove False positive:
- Non Maximum Suppression
- Filter by threshold


## Further improvements

- [ ] Adding Test Time Augmentation (TTA)
- [ ] Train a model robust to False Positives (Transformer based such as DETR)
- [ ] Train a YOLO architecture