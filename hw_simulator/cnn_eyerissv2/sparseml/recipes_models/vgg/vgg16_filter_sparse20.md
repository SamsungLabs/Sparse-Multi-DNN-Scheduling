---
# General Epoch/LR variables
num_epochs: &num_epochs 20
init_lr: &init_lr 0.0028
lr_step_epochs: &lr_step_epochs [7, 16]

# Pruning variables
pruning_start_epoch: &pruning_start_epoch 1.0
pruning_end_epoch: &pruning_end_epoch 8.0
pruning_update_frequency: &pruning_update_frequency 0.5
init_sparsity: &init_sparsity 0.05

prune_low_target_sparsity: &prune_low_target_sparsity 0.2
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.225
prune_high_target_sparsity: &prune_high_target_sparsity 0.25

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
  - !StructuredPruningModifier
    end_epoch: *pruning_end_epoch
    final_sparsity: *prune_low_target_sparsity
    init_sparsity: *init_sparsity
    inter_func: cubic
    leave_enabled: True
    log_types: __ALL__
    mask_type: filter
    params: ['sections.0.0.conv.weight', 'sections.0.1.conv.weight', 'sections.1.0.conv.weight', 'sections.1.1.conv.weight', 'sections.2.0.conv.weight', 'sections.2.1.conv.weight', 'sections.2.2.conv.weight', 'sections.3.0.conv.weight', 'sections.3.1.conv.weight', 'sections.3.2.conv.weight', 'sections.4.0.conv.weight', 'sections.4.1.conv.weight', 'sections.4.2.conv.weight']
    start_epoch: *pruning_start_epoch
    update_frequency: *pruning_update_frequency

  - !StructuredPruningModifier
    end_epoch: *pruning_end_epoch
    final_sparsity: *prune_high_target_sparsity
    init_sparsity: *init_sparsity
    inter_func: cubic
    leave_enabled: True
    log_types: __ALL__
    mask_type: filter
    params: ['classifier.mlp.0.weight', 'classifier.mlp.3.weight', 'classifier.mlp.6.weight']
    start_epoch: *pruning_start_epoch
    update_frequency: *pruning_update_frequency
---

# ResNet-50 imagenette Conservative Sparse

This recipe creates a sparse [ResNet-50](https://arxiv.org/abs/1512.03385) model that
achieves full recovery of its baseline accuracy on the imagenette dataset.
Training was done using 1 GPUs using a total training batch size of 128
using an SGD optimizer.

When running, adjust hyperparameters based on training environment and dataset.

## Training
The training script can be found at `sparseml/scripts/pytorch_vision.py`. 
Alternatively, a full walk-through notebook is located at `sparseml/notebooks/pytorch_classification.ipynb`.

*script command:*

```
python scripts/pytorch_vision.py train \
    --recipe-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenette/pruned-conservative?recipe_type=original \
    --pretrained True \
    --arch-key resnet50 \
    --dataset imagenette \
    --dataset-path /PATH/TO/IMAGENETTE  \
    --train-batch-size 128 --test-batch-size 256 \
    --loader-num-workers 8 \
    --model-tag resnet50-imagenette-pruned-conservative
```