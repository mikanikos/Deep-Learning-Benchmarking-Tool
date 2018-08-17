# Deep Learning Benchmarking Tool

Project work carried out at the Department of Computer Engineering (DISCA) of Technical University of Valencia (UPV) under the supervision of Professors Federico Silla Jiménez and Carlos Reaño González. It is a very simple suite which allows users to analyze the performance of different Deep Learning environments. It currently supports:

*  3 Metrics: average training speed, GPU memory and power usage
*  3 Datasets: ImageNet, CIFAR10, MNIST
*  5 Models: LeNet, AlexNet, ResNet50, GoogLeNet, VGG16
*  5 Frameworks: Caffe, TensorFlow, Pytorch, MXNet, CNTK
*  4 Hardware configurations for training: CPU, GPU and multi-GPU (as of now 2 and 4 GPUs only)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for benchmarking.

### Prerequisites

It is necessary to install the frameoworks you want to use, make sure they work in the correct way. Also CUDA installation is taken for granted.

### Dataset preparation

In order to run a benchmark, one or more datasetes must be downloaded and converted to the framework-specific format. Refer to the websites of each tool for the details. After that, the data path for each file has to be modified according to the new location, it takes just a sed command to change the default path /scratch/andpic/<datasetName>/<frameworkRecords> for every file in the entire project (<datasetName> could be "mnist", "cifar10" or "imageNet"; <frameworkRecords> could be "caffeRecords", "tfRecords", "pytorchRecords", "mxnetRecords" or "cntkRecords").
  
## Running benchmarks

Once the datasets have been set-up, run a benchmark using main.py (use -help for the parameters). 

Example:

```
python main.py -dataset imagenet -model alexnet -tool tensorflow -num_gpus 1 -gpu P100
```

### Benchmark report

The training session will be saved in two log-files in the folder logs, respectively a log_file with the main information about the training process and a power_log_file with the complete GPU monitoring.

## Notes

Please note that the most relevant networks' parameters can be changed directly in main.py (not the best way to do that but it is very immediate and less complicated than going deeper in every framework's file).

## Acknowledgments

The entire tool takes inspiration from other popular projects and it implements some of the core ideas of DLBENCH (https://github.com/hclhkbu/dlbench).

Networks' implementations and configurations have been taken from official sources and then modified:

*  https://github.com/NVIDIA/caffe for Caffe models
*  https://github.com/tensorflow/benchmarks for TensorFlow models
*  https://github.com/pytorch/examples for PyTorch models
*  https://github.com/apache/incubator-mxnet for MXNet models
*  https://github.com/Microsoft/CNTK for CNTK models
