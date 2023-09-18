---
# General Epoch/LR variables
num_epochs: &num_epochs 15.0
init_lr: &init_lr 0.001
lr_step_epochs: &lr_step_epochs [12]

# Pruning variables
pruning_start_epoch: &pruning_start_epoch 1.0
pruning_end_epoch: &pruning_end_epoch 10.0
pruning_update_frequency: &pruning_update_frequency 0.5
init_sparsity: &init_sparsity 0.05

prune_low_target_sparsity: &prune_low_target_sparsity 0.575
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.6
prune_high_target_sparsity: &prune_high_target_sparsity 0.625

# Modifiers
training_modifiers:
  - !EpochRangeModifier
    end_epoch: *num_epochs
    start_epoch: 0.0

  - !LearningRateModifier
    constant_logging: False
    end_epoch: -1.0
    init_lr: *init_lr
    log_types: __ALL__
    lr_class: MultiStepLR
    lr_kwargs: {'milestones': *lr_step_epochs, 'gamma': 0.1}
    start_epoch: 0.0
    update_frequency: -1.0

pruning_modifiers:
  - !GMPruningModifier
    params:
      - feature_extractor.1.0.conv1.weight
      - feature_extractor.1.1.conv1.weight
      - feature_extractor.1.1.conv2.weight
      - feature_extractor.2.0.conv1.weight
      - feature_extractor.2.0.conv2.weight
      - feature_extractor.3.0.conv1.weight
      - head.0.conv1.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_low_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - feature_extractor.1.0.conv2.weight
      - feature_extractor.2.0.identity.conv.weight
      - feature_extractor.2.1.conv1.weight
      - feature_extractor.2.1.conv2.weight
      - feature_extractor.3.1.conv1.weight
      - head.0.conv2.weight
      - head.1.conv2.weight
      - head.2.conv2.weight
      - predictor.location_predictor.3.weight
      - predictor.confidence_predictor.3.weight
      - head.3.conv2.weight
      - predictor.location_predictor.4.weight
      - predictor.confidence_predictor.4.weight
      - head.4.conv2.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_mid_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - feature_extractor.3.0.identity.conv.weight
      - feature_extractor.3.0.conv2.weight
      - feature_extractor.3.1.conv2.weight
      - predictor.location_predictor.0.weight
      - predictor.confidence_predictor.0.weight
      - predictor.confidence_predictor.1.weight
      - predictor.location_predictor.1.weight
      - predictor.location_predictor.2.weight
      - predictor.confidence_predictor.2.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_high_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
---

# SSD300-ResNet18 VOC Moderate Sparse

This recipe creates a sparse [SSD300-ResNet18](https://arxiv.org/abs/1512.02325) model that
achieves 99% recovery of its baseline accuracy on the VOC detection dataset.
Training was done using 1 GPU using a total training batch size of 64
using an SGD optimizer.

When running, adjust hyperparameters based on training environment and dataset.

## Training
The training script can be found at `sparseml/scripts/pytorch_vision.py`.
It runs the necessary pre-processing and loss function for training the SSD architecture.

*script command:*

```
python scripts/pytorch_vision.py train \
    --recipe-path zoo:cv/detection/ssd-resnet18_300/pytorch/sparseml/voc/pruned-moderate?recipe_type=original \
    --pretrained True \
    --arch-key ssd_resnet18 \
    --dataset voc_detection \
    --dataset-path /PATH/TO/IMAGENET  \
    --train-batch-size 64 --test-batch-size 128 \
    --loader-num-workers 6 \
    --model-tag ssd300_resnet18-imagenet-pruned-moderate
```