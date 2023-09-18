---
# General Epoch/LR variables
num_epochs: &num_epochs 20
init_lr: &init_lr 0.0028
lr_step_epochs: &lr_step_epochs [9, 15]

# Pruning variables
pruning_start_epoch: &pruning_start_epoch 0.0
pruning_end_epoch: &pruning_end_epoch 20.0
pruning_update_frequency: &pruning_update_frequency 0.4
init_sparsity: &init_sparsity 0.05

prune_low_target_sparsity: &prune_low_target_sparsity 0.2
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.225
prune_high_target_sparsity: &prune_high_target_sparsity 0.25


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
    params: ['sections.1.0.point.conv.weight', 'sections.1.1.point.conv.weight', 'sections.2.0.point.conv.weight']
    start_epoch: *pruning_start_epoch
    update_frequency: *pruning_update_frequency

  - !StructuredPruningModifier
    end_epoch: *pruning_end_epoch
    final_sparsity: *prune_mid_target_sparsity
    init_sparsity: *init_sparsity
    inter_func: cubic
    leave_enabled: True
    log_types: __ALL__
    mask_type: filter
    params: ['sections.2.1.point.conv.weight', 'sections.3.0.point.conv.weight', 'sections.3.1.point.conv.weight', 'sections.3.5.point.conv.weight']
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
    params: ['sections.3.2.point.conv.weight', 'sections.3.3.point.conv.weight', 'sections.3.4.point.conv.weight', 'sections.4.0.point.conv.weight', 'sections.4.1.point.conv.weight']
    start_epoch: *pruning_start_epoch
    update_frequency: *pruning_update_frequency
---

# MobileNet-V1 Pruned Conservative

This recipe creates a sparse [MobileNet-V1](https://arxiv.org/abs/1704.04861) model that achieves full recovery of its baseline accuracy on the ImageNet dataset.
Training was done using 4 GPUs using a total training batch size of 1024 using an SGD optimizer.

When running, adjust hyperparameters based on training environment and dataset.

## Training

To set up the training environment, follow the instructions on the [PyTorch SparseML integration README](https://github.com/neuralmagic/sparseml/tree/main/integrations/pytorch).
Using the given training script `vision.py` from the integration, the following command can be used to launch this recipe. 
Adjust the script command for your GPU device setup. 

*script command:*

```
python vision.py train \
    --recipe-path zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-conservative?recipe_type=original \
    --pretrained True \
    --arch-key mobilenet \
    --dataset imagenet \
    --dataset-path /PATH/TO/IMAGENET  \
    --train-batch-size 1024 --test-batch-size 2056 \
    --loader-num-workers 16 \
    --model-tag mobilenet-imagenet-pruned-conservative
```
