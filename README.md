# DeepLearningBenchmarkingTool
Project work carried out at the Department of Computer Engineering (DISCA) of Technical University of Valencia (UPV) under the supervision of Professors Federico Silla Jiménez and Carlos Reaño González. It is a very simple suite which allows users to analyze the performance of a specific set of Deep Learning parameters. It currently supports:

Metrics: average training speed, GPU memory and power usage

Datasets: ImageNet, CIFAR10, MNIST

Models: LeNet, AlexNet, ResNet50, GoogLeNet, VGG16

Frameworks: Caffe, TensorFlow, Pytorch, MXNet, CNTK

Hardware configurations for training: CPU, GPU and multi-GPU (up to 4 GPUS)

The entire tool takes inspiration from other popular projects and it implements some of the core ideas of DLBENCH (https://github.com/hclhkbu/dlbench), especially for acquiring GPU data and monitoring the training process.

First, in order to run a benchmark it is necessary to download the dataset you want to use and convert it to the framework-specific format. Refer to each tool folder for the details. After that, the data path for each file have to be modified according to the new location, it takes just a command with sed to change the default path /scratch/andpic/<datasetName>/<frameworkRecords> for every file in the entire project (<datasetName> could be "mnist", "cifar10" or "imageNet"; <frameworkRecords> could be "caffeRecords", "tfRecords", "pytorchRecords", "mxnetRecords" or "cntkRecords").

Once the dataset have been set-up, run a benchmark using main.py (use --help for the parameters).

Example: 
python main.py -dataset imagenet -model alexnet -tool tensorflow -num_gpus 1 -gpu P100

Please note that the most relevant networks' parameters can be changed directly in main.py (not the best way to do that but it is very immediate and less complicated than going deeper in every framework's files).
