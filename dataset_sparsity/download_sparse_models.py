from sparsezoo import Model

# SSD download sparsity
stub = "zoo:cv/detection/ssd-resnet50_300/pytorch/sparseml/coco/pruned-moderate"
model = Model(stub, download_path="./models/ssd_prunned")
model.download()

# ResNet download sparsity
stub = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none"
model = Model(stub, download_path="./models/resnet_pruned_nonquant")
model.download()

# Mobilenetv1 download sparsity
stub = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
model = Model(stub, download_path="./models/mobilenetv1_prunned")
model.download()

# Mobilenetv2 download original
stub = "zoo:cv/classification/mobilenet_v2-1.0/pytorch/torchvision/imagenet/base-none"
model = Model(stub, download_path="./models/mobilenetv2")
model.download()

# VGG download with unstructured sparsity
stub = "zoo:cv/classification/vgg-16/pytorch/sparseml/imagenet/pruned-moderate"
model = Model(stub, download_path="./models/vgg16_prunned")
model.download()

# Inceptionv3 download sparsity
stub = "zoo:cv/classification/inception_v3/pytorch/sparseml/imagenet/pruned-moderate"
model = Model(stub, download_path="./models/inceptionv3_prunned")
model.download()
