# Constants for quantization functions
DITHERED_8_BIT_QUANTIZATION_SCALE = 1  # 0.5 0.05

BACKWARD_QUANTIZATION_GROUP_SIZE = 2048
BACKWARD_QUANTIZATION_GROUP_SHIFT_AMOUNT = 11

STD_UPPER_BOUND = 127
STD_LOWER_BOUND = -128

FORWARD_QUANTIZATION_GROUP_SIZE = 2048
FORWARD_QUANTIZATION_GROUP_SHIFT_AMOUNT = 11

SAWB_UPPER_BOUND = 127
SAWB_LOWER_BOUND = -128
SAWB_COEFFICIENT_1 = 12.1
SAWB_COEFFICIENT_2 = 12.2

PARALLEL = True
GROUPED = False

# Training constants

# CIFAR-10 constants
CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)

# CIFAR-100 constants
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# MNIST constants
MNIST_TRAIN_MEAN = (0.5,)
MNIST_TRAIN_STD = (0.5,)

# Stanford Cars
CARS_TRAIN_MEAN = (0.485, 0.456, 0.406)
CARS_TRAIN_STD = (0.229, 0.224, 0.225)

# Caltech101
CAL101_TRAIN_MEAN = (0.485, 0.456, 0.406)
CAL101_TRAIN_STD = (0.229, 0.224, 0.225)

# Caltech256
CAL256_TRAIN_MEAN = (0.485, 0.456, 0.406)
CAL256_TRAIN_STD = (0.229, 0.224, 0.225)

# Plant Disease
PLANT_DISEASE_TRAIN_MEAN = (0.485, 0.456, 0.406)
PLANT_DISEASE_TRAIN_STD = (0.229, 0.224, 0.225)


