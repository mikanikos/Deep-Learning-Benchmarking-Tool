import argparse
import sys,os,time
import subprocess
import get_gpu_usage as cgp
from threading import Thread
import multiprocessing

parser = argparse.ArgumentParser(description=' Deep Learning Benchmarking tool')

parser.add_argument('-tool', type=str, help='Tool to use for training: caffe, tensorflow, pytorch, mxnet, cntk')
parser.add_argument('-gpu', type=str, help='Name of the gpu, ex. P100')
parser.add_argument('-num_gpus', type=str, help='Number of gpus to use for training, 0 for CPU')
parser.add_argument('-dataset', type=str, help='Dataset to use for training: mnist, cifar10, imagenet')
parser.add_argument('-model', type=str, help='Model to train: lenet, alexnet, resnet50, googlenet, vgg16')

args = parser.parse_args()

log_file = args.tool + "_" + args.num_gpus + "-" + args.gpu + "_" + args.dataset + "_" + args.model + "_" + str(time.ctime()) + ".log"
log_file = log_file.replace(" ","_")

tool = args.tool
num_gpus = args.num_gpus
dataset = args.dataset
model = args.model

if dataset == "imagenet":
	if model == "alexnet":
		if tool == "caffe":
			if num_gpus == "0":
				cmd = "caffe train -solver=caffe/imagenet/alexnet/solver.prototxt"
			if num_gpus == "1":
				cmd = "caffe train -solver=caffe/imagenet/alexnet/solver.prototxt -gpu=0"
			if num_gpus == "2":
				cmd = "caffe train -solver=caffe/imagenet/alexnet/solver_2gpu.prototxt -gpu=0,1"
			if num_gpus == "4":	
				cmd = "caffe train -solver=caffe/imagenet/alexnet/solver_4gpu.prototxt -gpu=0,1,2,3"

		if tool == "tensorflow":
			if num_gpus == "0":
				cmd = "python tf_cnn_benchmarks.py --data_format=NHWC --batch_size=128 --model=alexnet --device=cpu --init_learning_rate=0.01 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "1":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=128 --model=alexnet --num_gpus=1 --init_learning_rate=0.01 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "2":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=128 --model=alexnet --num_gpus=2 --init_learning_rate=0.01 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "4":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=128 --model=alexnet --num_gpus=4 --init_learning_rate=0.01 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"

		if tool == "pytorch":
			if num_gpus == "0":
				cmd = "python main.py -a alexnet -j 40 --epochs 1 -b 128 --lr 0.01 --no-cuda /scratch/andpic/imageNet"
			if num_gpus == "1":
				cmd = "python main.py -a alexnet -j 40 --epochs 1 -b 128 --lr 0.01 /scratch/andpic/imageNet"
			if num_gpus == "2":
				cmd = "python main.py -a alexnet -j 40 --epochs 1 -b 128 --lr 0.01 /scratch/andpic/imageNet"
			if num_gpus == "4":	
				cmd = "python main.py -a alexnet -j 40 --epochs 1 -b 128 --lr 0.01 /scratch/andpic/imageNet"

		if tool == "mxnet":
			if num_gpus == "0":
				cmd = "python train_imagenet.py --gpus '' --network alexnet --batch-size 128 --num-epochs 1 --data-nthreads 40 --lr 0.01 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec"
			if num_gpus == "1":
				cmd = "python train_imagenet.py --gpus 0 --network alexnet --batch-size 128 --num-epochs 1 --data-nthreads 40 --lr 0.01 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec"
			if num_gpus == "2":
				cmd = "python train_imagenet.py --gpus 0,1 --network alexnet --batch-size 128 --num-epochs 1 --data-nthreads 40 --lr 0.01 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec"
			if num_gpus == "4":	
				cmd = "python train_imagenet.py --gpus 0,1,2,3 --network alexnet --batch-size 128 --num-epochs 1 --data-nthreads 40 --lr 0.01 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec"

		if tool == "cntk":
			if num_gpus == "0":
				cmd = "python AlexNet_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 38400 -m 128 -r"
			if num_gpus == "1":
				cmd = "mpiexec --npernode 1 python AlexNet_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 38400 -m 128 -r"
			if num_gpus == "2":
				cmd = "mpiexec --npernode 2 python AlexNet_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 38400 -m 128 -r"
			if num_gpus == "4":	
				cmd = "mpiexec --npernode 4 python AlexNet_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 38400 -m 128 -r"

	if model == "resnet50":
		if tool == "caffe":	
			if num_gpus == "0":
				cmd = "caffe train -solver=caffe/imagenet/resnet50/solver.prototxt"
			if num_gpus == "1":
				cmd = "caffe train -solver=caffe/imagenet/resnet50/solver.prototxt -gpu=0"
			if num_gpus == "2":
				cmd = "caffe train -solver=caffe/imagenet/resnet50/solver_2gpu.prototxt -gpu=0,1"
			if num_gpus == "4":	
				cmd = "caffe train -solver=caffe/imagenet/resnet50/solver_4gpu.prototxt -gpu=0,1,2,3"

		if tool == "tensorflow":
			if num_gpus == "0":
				cmd = "python tf_cnn_benchmarks.py --data_format=NHWC --batch_size=8 --model=resnet50 --device=cpu --init_learning_rate=0.1 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "1":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=8 --model=resnet50 --num_gpus=1 --init_learning_rate=0.1 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "2":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=8 --model=resnet50 --num_gpus=2 --init_learning_rate=0.1 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "4":	
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=8 --model=resnet50 --num_gpus=4 --init_learning_rate=0.1 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"

		if tool == "pytorch":
			if num_gpus == "0":
				cmd = "python main.py -a resnet50 -j 40 --epochs 1 -b 8 --lr 0.1 --no-cuda /scratch/andpic/imageNet"
			if num_gpus == "1":
				cmd = "python main.py -a resnet50 -j 40 --epochs 1 -b 8 --lr 0.1 /scratch/andpic/imageNet"
			if num_gpus == "2":
				cmd = "python main.py -a resnet50 -j 40 --epochs 1 -b 8 --lr 0.1 /scratch/andpic/imageNet"
			if num_gpus == "4":	
				cmd = "python main.py -a resnet50 -j 40 --epochs 1 -b 8 --lr 0.1 /scratch/andpic/imageNet"

		if tool == "mxnet":
			if num_gpus == "0":
				cmd = "python train_imagenet.py --gpus '' --network resnet --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.1 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec --num-layers 50"
			if num_gpus == "1":
				cmd = "python train_imagenet.py --gpus 0 --network resnet --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.1 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec --num-layers 50"
			if num_gpus == "2":
				cmd = "python train_imagenet.py --gpus 0,1 --network resnet --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.1 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec --num-layers 50"
			if num_gpus == "4":	
				cmd = "python train_imagenet.py --gpus 0,1,2,3 --network resnet --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.1 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec --num-layers 50"

		if tool == "cntk":
			if num_gpus == "0":
				cmd = "python TrainResNet_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n resnet50 -s 24 -es 2400 -e 1 -r"
			if num_gpus == "1":
				cmd = "mpiexec --npernode 1 python TrainResNet_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n resnet50 -s 24 -es 2400 -e 1 -r"
			if num_gpus == "2":
				cmd = "mpiexec --npernode 2 python TrainResNet_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n resnet50 -s 24 -es 2400 -e 1 -r"
			if num_gpus == "4":	
				cmd = "mpiexec --npernode 4 python TrainResNet_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n resnet50 -s 24 -es 2400 -e 1 -r"

	if model == "googlenet":
		if tool == "caffe":	
			if num_gpus == "0":
				cmd = "caffe train -solver=caffe/imagenet/googlenet/solver.prototxt"
			if num_gpus == "1":
				cmd = "caffe train -solver=caffe/imagenet/googlenet/solver.prototxt -gpu=0"
			if num_gpus == "2":
				cmd = "caffe train -solver=caffe/imagenet/googlenet/solver_2gpu.prototxt -gpu=0,1"
			if num_gpus == "4":	
				cmd = "caffe train -solver=caffe/imagenet/googlenet/solver_4gpu.prototxt -gpu=0,1,2,3"

		if tool == "tensorflow":
			if num_gpus == "0":
				cmd = "python tf_cnn_benchmarks.py --data_format=NHWC --batch_size=8 --model=googlenet --device=cpu --init_learning_rate=0.1 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "1":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=8 --model=googlenet --num_gpus=1 --init_learning_rate=0.1 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "2":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=8 --model=googlenet --num_gpus=2 --init_learning_rate=0.1 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "4":	
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=8 --model=googlnet --num_gpus=4 --init_learning_rate=0.1 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"

		if tool == "pytorch":
			if num_gpus == "0":
				cmd = ""
			if num_gpus == "1":
				cmd = ""
			if num_gpus == "2":
				cmd = ""
			if num_gpus == "4":	
				cmd = ""

		if tool == "mxnet":
			if num_gpus == "0":
				cmd = "python train_imagenet.py --gpus '' --network googlenet --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.1 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec"
			if num_gpus == "1":
				cmd = "python train_imagenet.py --gpus 0 --network googlenet --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.1 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec"
			if num_gpus == "2":
				cmd = "python train_imagenet.py --gpus 0,1 --network googlenet --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.1 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec"
			if num_gpus == "4":	
				cmd = "python train_imagenet.py --gpus 0,1,2,3 --network googlenet --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.1 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec"


		if tool == "cntk":
			if num_gpus == "0":
				cmd = "python BN_Inception_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 2400 -m 8 -r"
			if num_gpus == "1":
				cmd = "mpiexec --npernode 1 python BN_Inception_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 2400 -m 8 -r"
			if num_gpus == "2":
				cmd = "mpiexec --npernode 2 python BN_Inception_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 2400 -m 8 -r"
			if num_gpus == "4":	
				cmd = "mpiexec --npernode 4 python BN_Inception_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 2400 -m 8 -r"

	if model == "vgg16":
		if tool == "caffe":	
			if num_gpus == "0":
				cmd = "caffe train -solver=caffe/imagenet/vgg16/solver.prototxt"
			if num_gpus == "1":
				cmd = "caffe train -solver=caffe/imagenet/vgg16/solver.prototxt -gpu=0"
			if num_gpus == "2":
				cmd = "caffe train -solver=caffe/imagenet/vgg16/solver_2gpu.prototxt -gpu=0,1"
			if num_gpus == "4":	
				cmd = "caffe train -solver=caffe/imagenet/vgg16/solver_4gpu.prototxt -gpu=0,1,2,3"

		if tool == "tensorflow":
			if num_gpus == "0":
				cmd = "python tf_cnn_benchmarks.py --data_format=NHWC --batch_size=8 --model=vgg16 --device=cpu --init_learning_rate=0.01 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "1":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=8 --model=vgg16 --num_gpus=1 --init_learning_rate=0.01 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "2":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=8 --model=vgg16 --num_gpus=2 --init_learning_rate=0.01 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"
			if num_gpus == "4":	
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=8 --model=vgg16 --num_gpus=4 --init_learning_rate=0.01 --num_batches=300 --data_dir=/scratch/andpic/imageNet/tfRecords --data_name=imagenet --num_intra_threads 40 --num_inter_threads 2"

		if tool == "pytorch":
			if num_gpus == "0":
				cmd = "python main.py -a vgg16 -j 40 --epochs 1 -b 8 --lr 0.01 --no-cuda /scratch/andpic/imageNet"
			if num_gpus == "1":
				cmd = "python main.py -a vgg16 -j 40 --epochs 1 -b 8 --lr 0.01 /scratch/andpic/imageNet"
			if num_gpus == "2":
				cmd = "python main.py -a vgg16 -j 40 --epochs 1 -b 8 --lr 0.01 /scratch/andpic/imageNet"
			if num_gpus == "4":	
				cmd = "python main.py -a vgg16 -j 40 --epochs 1 -b 8 --lr 0.01 /scratch/andpic/imageNet"

		if tool == "mxnet":
			if num_gpus == "0":
				cmd = "python train_imagenet.py --gpus '' --network vgg --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.01 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec --num-layers 16"
			if num_gpus == "1":
				cmd = "python train_imagenet.py --gpus 0 --network vgg --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.01 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec --num-layers 16"
			if num_gpus == "2":
				cmd = "python train_imagenet.py --gpus 0,1 --network vgg --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.01 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec --num-layers 16"
			if num_gpus == "4":	
				cmd = "python train_imagenet.py --gpus 0,1,2,3 --network vgg --batch-size 8 --num-epochs 1 --data-nthreads 40 --lr 0.01 --data-train=/scratch/andpic/imageNet/mxnetRecords/mxdata.rec --num-layers 16"

		if tool == "cntk":
			if num_gpus == "0":
				cmd = "python VGG16_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 2400 -m 8 -r"
			if num_gpus == "1":
				cmd = "mpiexec --npernode 1 python VGG16_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 2400 -m 8 -r"
			if num_gpus == "2":
				cmd = "mpiexec --npernode 2 python VGG16_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 2400 -m 8 -r"
			if num_gpus == "4":	
				cmd = "mpiexec --npernode 4 python VGG16_ImageNet_Distributed.py -datadir /scratch/andpic/imageNet/train -n 1 -e 2400 -m 8 -r"

if dataset == "cifar10":
	if model == "alexnet":
		if tool == "caffe":
			if num_gpus == "0":
				cmd = "caffe train -solver=caffe/cifar10/alexnet/solver.prototxt"
			if num_gpus == "1":
				cmd = "caffe train -solver=caffe/cifar10/alexnet/solver.prototxt -gpu=0"
			if num_gpus == "2":
				cmd = "caffe train -solver=caffe/cifar10/alexnet/solver_2gpu.prototxt -gpu=0,1"
			if num_gpus == "4":	
				cmd = "caffe train -solver=caffe/cifar10/alexnet/solver_4gpu.prototxt -gpu=0,1,2,3"

		if tool == "tensorflow":
			if num_gpus == "0":
				cmd = "python tf_cnn_benchmarks.py --data_format=NHWC --batch_size=128 --model=alexnet --device=cpu --init_learning_rate=0.01 --num_epochs=1 --data_dir=/scratch/andpic/cifar10/cifar-10-batches-py --data_name=cifar10 --num_intra_threads 24 --num_inter_threads 2"
			if num_gpus == "1":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=128 --model=alexnet --num_gpus=1 --init_learning_rate=0.01 --num_epochs=1 --data_dir=/scratch/andpic/cifar10/cifar-10-batches-py --data_name=cifar10 --num_intra_threads 24 --num_inter_threads 2"
			if num_gpus == "2":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=128 --model=alexnet --num_gpus=2 --init_learning_rate=0.01 --num_epochs=1 --data_dir=/scratch/andpic/cifar10/cifar-10-batches-py --data_name=cifar10 --num_intra_threads 24 --num_inter_threads 2"
			if num_gpus == "4":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=128 --model=alexnet --num_gpus=4 --init_learning_rate=0.01 --num_epochs=1 --data_dir=/scratch/andpic/cifar10/cifar-10-batches-py --data_name=cifar10 --num_intra_threads 24 --num_inter_threads 2"

		if tool == "pytorch":
			if num_gpus == "0":
				cmd = "python cifar.py --epochs 1 --lr 0.01 -j 24 --train-batch 128 -d cifar10 --no-cuda -a alexnet"
			if num_gpus == "1":
				cmd = "python cifar.py --epochs 1 --lr 0.01 -j 24 --train-batch 128 -d cifar10 --gpu-id 0 -a alexnet"
			if num_gpus == "2":
				cmd = "python cifar.py --epochs 1 --lr 0.01 -j 24 --train-batch 64 -d cifar10 --gpu-id 0,1 -a alexnet"
			if num_gpus == "4":	
				cmd = "python cifar.py --epochs 1 --lr 0.01 -j 24 --train-batch 32 -d cifar10 --gpu-id 0,1,2,3 -a alexnet"

		if tool == "mxnet":
			if num_gpus == "0":
				cmd = "python train_cifar10.py --gpus '' --network alexnet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.01"
			if num_gpus == "1":
				cmd = "python train_cifar10.py --gpus 0 --network alexnet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.01"
			if num_gpus == "2":
				cmd = "python train_cifar10.py --gpus 0,1 --network alexnet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.01"
			if num_gpus == "4":	
				cmd = "python train_cifar10.py --gpus 0,1,2,3 --network alexnet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.01"

		if tool == "cntk":
			if num_gpus == "0":
				cmd = ""
			if num_gpus == "1":
				cmd = ""
			if num_gpus == "2":
				cmd = ""
			if num_gpus == "4":	
				cmd = ""

	if model == "resnet50":
		if tool == "caffe":	
			if num_gpus == "0":
				cmd = "caffe train -solver=caffe/cifar10/resnet50/solver.prototxt"
			if num_gpus == "1":
				cmd = "caffe train -solver=caffe/cifar10/resnet50/solver.prototxt -gpu=0"
			if num_gpus == "2":
				cmd = "caffe train -solver=caffe/cifar10/resnet50/solver_2gpu.prototxt -gpu=0,1"
			if num_gpus == "4":	
				cmd = "caffe train -solver=caffe/cifar10/resnet50/solver_4gpu.prototxt -gpu=0,1,2,3"

		if tool == "tensorflow":
			if num_gpus == "0":
				cmd = "python tf_cnn_benchmarks.py --data_format=NHWC --batch_size=128 --model=resnet56 --device=cpu --init_learning_rate=0.01 --num_epochs=1 --data_dir=/scratch/andpic/cifar10/cifar-10-batches-py --data_name=cifar10 --num_intra_threads 24 --num_inter_threads 2"
			if num_gpus == "1":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=128 --model=resnet56 --num_gpus=1 --init_learning_rate=0.01 --num_epochs=1 --data_dir=/scratch/andpic/cifar10/cifar-10-batches-py --data_name=cifar10 --num_intra_threads 24 --num_inter_threads 2"
			if num_gpus == "2":
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=128 --model=resnet56 --num_gpus=2 --init_learning_rate=0.01 --num_epochs=1 --data_dir=/scratch/andpic/cifar10/cifar-10-batches-py --data_name=cifar10 --num_intra_threads 24 --num_inter_threads 2"
			if num_gpus == "4":	
				cmd = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=128 --model=resnet56 --num_gpus=4 --init_learning_rate=0.01 --num_epochs=1 --data_dir=/scratch/andpic/cifar10/cifar-10-batches-py --data_name=cifar10 --num_intra_threads 24 --num_inter_threads 2"

		if tool == "pytorch":
			if num_gpus == "0":
				cmd = "python cifar.py --epochs 1 --lr 0.1 -j 24 --train-batch 128 -d cifar10 --no-cuda -a resnet --depth 50"
			if num_gpus == "1":
				cmd = "python cifar.py --epochs 1 --lr 0.1 -j 24 --train-batch 128 -d cifar10 --gpu-id 0 -a resnet --depth 50"
			if num_gpus == "2":
				cmd = "python cifar.py --epochs 1 --lr 0.1 -j 24 --train-batch 64 -d cifar10 --gpu-id 0,1 -a resnet --depth 50"
			if num_gpus == "4":	
				cmd = "python cifar.py --epochs 1 --lr 0.1 -j 24 --train-batch 32 -d cifar10 --gpu-id 0,1,2,3 -a resnet --depth 50"

		if tool == "mxnet":
			if num_gpus == "0":
				cmd = "python train_cifar10.py --gpus '' --network resnet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.1 --num-layers 50"
			if num_gpus == "1":
				cmd = "python train_cifar10.py --gpus 0 --network resnet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.1 --num-layers 50"
			if num_gpus == "2":
				cmd = "python train_cifar10.py --gpus 0,1 --network resnet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.1 --num-layers 50"
			if num_gpus == "4":	
				cmd = "python train_cifar10.py --gpus 0,1,2,3 --network resnet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.1 --num-layers 50"

		if tool == "cntk":
			if num_gpus == "0":
				cmd = "python TrainResNet_CIFAR10_Distributed.py -n resnet20 -s 24 -datadir /scratch/andpic/cifar10/cntkRecords -e 1 -r"
			if num_gpus == "1":
				cmd = "mpiexec --npernode 1 python TrainResNet_CIFAR10_Distributed.py -n resnet20 -s 24 -datadir /scratch/andpic/cifar10/cntkRecords -e 1 -r"
			if num_gpus == "2":
				cmd = "mpiexec --npernode 2 python TrainResNet_CIFAR10_Distributed.py -n resnet20 -s 24 -datadir /scratch/andpic/cifar10/cntkRecords -e 1 -r"
			if num_gpus == "4":	
				cmd = "mpiexec --npernode 4 python TrainResNet_CIFAR10_Distributed.py -n resnet20 -s 24 -datadir /scratch/andpic/cifar10/cntkRecords -e 1 -r"

	if model == "googlenet":
		if tool == "caffe":	
			if num_gpus == "0":
				cmd = "caffe train -solver=caffe/cifar10/googlenet/solver.prototxt"
			if num_gpus == "1":
				cmd = "caffe train -solver=caffe/cifar10/googlenet/solver.prototxt -gpu=0"
			if num_gpus == "2":
				cmd = "caffe train -solver=caffe/cifar10/googlenet/solver_2gpu.prototxt -gpu=0,1"
			if num_gpus == "4":	
				cmd = "caffe train -solver=caffe/cifar10/googlenet/solver_4gpu.prototxt -gpu=0,1,2,3"

		if tool == "tensorflow":
			if num_gpus == "0":
				cmd = ""
			if num_gpus == "1":
				cmd = ""
			if num_gpus == "2":
				cmd = ""
			if num_gpus == "4":	
				cmd = ""

		if tool == "pytorch":
			if num_gpus == "0":
				cmd = ""
			if num_gpus == "1":
				cmd = ""
			if num_gpus == "2":
				cmd = ""
			if num_gpus == "4":	
				cmd = ""

		if tool == "mxnet":
			if num_gpus == "0":
				cmd = "python train_cifar10.py --gpus '' --network googlenet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.1"
			if num_gpus == "1":
				cmd = "python train_cifar10.py --gpus 0 --network googlenet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.1"
			if num_gpus == "2":
				cmd = "python train_cifar10.py --gpus 0,1 --network googlenet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.1"
			if num_gpus == "4":	
				cmd = "python train_cifar10.py --gpus 0,1,2,3 --network googlenet --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.1"

		if tool == "cntk":
			if num_gpus == "0":
				cmd = "python BN_Inception_CIFAR10_Distributed.py -s 24 -datadir /scratch/andpic/cifar10/cntkRecords -n 1 -r -m 128"
			if num_gpus == "1":
				cmd = "mpiexec --npernode 1 python BN_Inception_CIFAR10_Distributed.py -s 24 -datadir /scratch/andpic/cifar10/cntkRecords -n 1 -r -m 128"
			if num_gpus == "2":
				cmd = "mpiexec --npernode 2 python BN_Inception_CIFAR10_Distributed.py -s 24 -datadir /scratch/andpic/cifar10/cntkRecords -n 1 -r -m 128"
			if num_gpus == "4":	
				cmd = "mpiexec --npernode 4 python BN_Inception_CIFAR10_Distributed.py -n resnet20 -s 24 -datadir /scratch/andpic/cifar10/cntkRecords -n 1 -r -m 128"

	if model == "vgg16":
		if tool == "caffe":	
			if num_gpus == "0":
				cmd = "caffe train -solver=caffe/cifar10/vgg16/solver.prototxt"
			if num_gpus == "1":
				cmd = "caffe train -solver=caffe/cifar10/vgg16/solver.prototxt -gpu=0"
			if num_gpus == "2":
				cmd = "caffe train -solver=caffe/cifar10/vgg16/solver_2gpu.prototxt -gpu=0,1"
			if num_gpus == "4":	
				cmd = "caffe train -solver=caffe/cifar10/vgg16/solver_4gpu.prototxt -gpu=0,1,2,3"

		if tool == "tensorflow":
			if num_gpus == "0":
				cmd = ""
			if num_gpus == "1":
				cmd = ""
			if num_gpus == "2":
				cmd = ""
			if num_gpus == "4":	
				cmd = ""

		if tool == "pytorch":
			if num_gpus == "0":
				cmd = "python cifar.py --epochs 1 --lr 0.01 -j 24 --train-batch 128 -d cifar10 --no-cuda -a vgg16"
			if num_gpus == "1":
				cmd = "python cifar.py --epochs 1 --lr 0.01 -j 24 --train-batch 128 -d cifar10 --gpu-id 0 -a vgg16"
			if num_gpus == "2":
				cmd = "python cifar.py --epochs 1 --lr 0.01 -j 24 --train-batch 64 -d cifar10 --gpu-id 0,1 -a vgg16"
			if num_gpus == "4":	
				cmd = "python cifar.py --epochs 1 --lr 0.01 -j 24 --train-batch 32 -d cifar10 --gpu-id 0,1,2,3 -a vgg16"

		if tool == "mxnet":
			if num_gpus == "0":
				cmd = "python train_cifar10.py --gpus '' --network vgg --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.01 --num-layers 16"
			if num_gpus == "1":
				cmd = "python train_cifar10.py --gpus 0 --network vgg --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.01 --num-layers 16"
			if num_gpus == "2":
				cmd = "python train_cifar10.py --gpus 0,1 --network vgg --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.01 --num-layers 16"
			if num_gpus == "4":	
				cmd = "python train_cifar10.py --gpus 0,1,2,3 --network vgg --batch-size 128 --num-epochs 1 --data-nthreads 24 --lr 0.01 --num-layers 16"

		if tool == "cntk":
			if num_gpus == "0":
				cmd = ""
			if num_gpus == "1":
				cmd = ""
			if num_gpus == "2":
				cmd = ""
			if num_gpus == "4":	
				cmd = ""	


if dataset == "mnist":
	if model == "lenet":
		if tool == "caffe":	
			if num_gpus == "0":
				cmd = "caffe train -solver=caffe/mnist/lenet/lenet_solver.prototxt"
			if num_gpus == "1":
				cmd = "caffe train -solver=caffe/mnist/lenet/lenet_solver.prototxt -gpu=0"
			if num_gpus == "2":
				cmd = "caffe train -solver=caffe/mnist/lenet/lenet_solver_2gpu.prototxt -gpu=0,1"
			if num_gpus == "4":	
				cmd = "caffe train -solver=caffe/mnist/lenet/lenet_solver_4gpu.prototxt -gpu=0,1,2,3"

		if tool == "tensorflow":
			if num_gpus == "0":
				cmd = "python train_lenet.py"
			if num_gpus == "1":
				cmd = "python train_lenet.py"
			if num_gpus == "2":
				cmd = "python train_lenet_2gpu.py"
			if num_gpus == "4":	
				cmd = "python train_lenet_4gpu.py"

		if tool == "pytorch":
			if num_gpus == "0":
				cmd = "python main_mnist.py --batch-size 64 --epochs 1 --lr 0.01 --no-cuda"
			if num_gpus == "1":
				cmd = "python main_mnist.py --batch-size 64 --epochs 1 --lr 0.01"
			if num_gpus == "2":
				cmd = "python main_mnist.py --batch-size 64 --epochs 1 --lr 0.01"
			if num_gpus == "4":
				cmd = "python main_mnist.py --batch-size 64 --epochs 1 --lr 0.01"

		if tool == "mxnet":
			if num_gpus == "0":
				cmd = "python train_mnist.py --gpus '' --network lenet --batch-size 64 --num-epochs 1 --lr 0.01"
			if num_gpus == "1":
				cmd = "python train_mnist.py --gpus 0 --network lenet --batch-size 64 --num-epochs 1 --lr 0.01"
			if num_gpus == "2":
				cmd = "python train_mnist.py --gpus 0,1 --network lenet --batch-size 64 --num-epochs 1 --lr 0.01"
			if num_gpus == "4":	
				cmd = "python train_mnist.py --gpus 0,1,2,3 --network lenet --batch-size 64 --num-epochs 1 --lr 0.01"

		if tool == "cntk":
			if num_gpus == "0":
				cmd = "python lenet.py"
			if num_gpus == "1":
				cmd = "python lenet.py"
			if num_gpus == "2":
				cmd = "python lenet.py"
			if num_gpus == "4":	
				cmd = "python lenet.py"

if num_gpus == "0":
	os.system("export CUDA_VISIBLE_DEVICES=")

if num_gpus == "1":
	os.system("export CUDA_VISIBLE_DEVICES=0")

if num_gpus == "2":
	os.system("export CUDA_VISIBLE_DEVICES=0,1")	

if num_gpus == "4":
	os.system("export CUDA_VISIBLE_DEVICES=0,1,2,3")


root_path = os.path.dirname(os.path.abspath(__file__))
tool_path = root_path + "/" + args.tool

if args.tool != "caffe":
    os.chdir(tool_path)

if args.tool == "cntk":
    os.chdir(tool_path + "/" + args.model)

power_log_file = '%s/logs/power_%s' % (root_path, log_file)

print ""

os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())

try:
    thread = Thread(target = cgp.start_collecting_gpu_power, args = (cmd, power_log_file))
    t = time.time()
    thread.start()
    os.system(cmd)
    t = time.time() - t
except Exception as e:
	print ("Benchmark failed or interruped with " + cmd)

os.chdir(root_path)
	
if ".log" not in log_file:
	log_file += ".log"
log_path = os.getcwd() + "/" + log_file

time.sleep(10)
power, mem = cgp.get_average_gpu_power_and_mem(args.gpu, power_log_file)

if ".log" not in log_file:
        log_file += ".log"
log_path = os.getcwd() + "/" + log_file

with open(log_path, "a") as logFile:
	logFile.write("Total time: " + str(t) + "\n")
	logFile.write("Average power: " + str(power) + "\n")
	logFile.write("Average memory: " + str(mem) + "\n")
	logFile.write("cmd: " + cmd + "\n")
os.system("mv " + log_path + " logs")

print ('')
os.system("cat " + "logs/" + log_file)
