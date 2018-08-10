import argparse
import sys,os,time
import subprocess
import get_gpu_usage as cgp
from threading import Thread
import multiprocessing

parser = argparse.ArgumentParser(description='Benchmark deep learning tools')

#parser.add_argument('-descr', type=str, help='Command to benchmark')
#parser.add_argument('-tool', type=str, help='Type of command')
#parser.add_argument('-command', type=str, help='Type of command')
parser.add_argument('-tool', type=str, help='Tool to use for training: caffe, tensorflow, pytorch, mxnet, cntk')
parser.add_argument('-gpu', type=str, help='Name of the gpu')
parser.add_argument('-num_gpus', type=str, help='Number of gpus to use for training, 0 for CPU')
parser.add_argument('-dataset', type=str, help='Dataset to use for training: mnist, cifar10, imagenet')
parser.add_argument('-model', type=str, help='Model to train: lenet, alexnet, resnet50, googlenet, vgg16')

args = parser.parse_args()


#caffe_cmd = "caffe train -solver=solver.prototxt"
#tensorflow_cmd = "python tf_cnn_benchmarks.py --optimizer=momentum --variable_update=replicated --nodistortions --gradient_repacking=8 --num_epochs=1 --weight_decay=1e-4 --data_name=" + args.dataset + " --use_fp16 --num_intra_threads 24 --num_inter_threads 2"
#pytorch_cmd = "python main.py -j 24 --epochs 1"
#mxnet_cmd = "python train_" + args.dataset + ".py --image-shape 3,299,299 --num-epochs 1 --worker_count 24 --data-train=/scratch/andpic/imageNet/train --data-val=/scratch/andpic/imageNet/train"
#cntk_cmd = "python run.py -datadir=/scratch/andpic/imageNet/train -n 1"

#log_file = args.descr + "_" + str(time.ctime()) + ".log"
log_file = args.tool + "_" + args.num_gpus + "-" + args.gpu + "_" + args.dataset + "_" + args.model + "_" + str(time.ctime()) + ".log"
log_file = log_file.replace(" ","_")

cmd = raw_input("Command to benchmark: ")
#cmd = args.command       

#print cmd

root_path = os.path.dirname(os.path.abspath(__file__))
tool_path = root_path + "/" + args.tool

if args.tool != "caffe":
    os.chdir(tool_path)

if args.tool == "cntk":
    os.chdir(tool_path + "/" + args.model)

power_log_file = '%s/logs/power_%s' % (root_path, log_file)

#run_cmd = "python2.7 run_bench.py -command=\"" + cmd + "\" -log=" + log_file + " -tool=" + args.tool

print ""
#print run_cmd

os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())

try:
    thread = Thread(target = cgp.start_collecting_gpu_power, args = (cmd, power_log_file))
    t = time.time()
    thread.start()
    os.system(cmd)
    #result = subprocess.check_output(run_cmd, shell=True)
    t = time.time() - t

#	if ".log" not in log_file:
#            log_file += ".log"
#	log_path = os.getcwd() + "/" + log_file
	
#	power, mem = cgp.get_average_gpu_power_and_mem("K20m", power_log_file)
	
#	with open(log_path, "a") as logFile:
#            logFile.write("Total time: " + str(t) + "\n")
#            logFile.write("Average power: " + str(power) + "\n")
#            logFile.write("Average memory: " + str(mem) + "\n")
#            logFile.write("cmd: " + cmd + "\n")
#	os.system("mv " + log_path + " logs")

#	os.system("cat " + "logs/" + log_file)	
#	os.system("cat " + power_log_file)
except Exception as e:
	print ("Benchmark failed or interruped with " + cmd)
	#os.system("cat " + root_path + "/logs/" + log_file)

os.chdir(root_path)
	
if ".log" not in log_file:
	log_file += ".log"
log_path = os.getcwd() + "/" + log_file

#print(subprocess.check_output("python tools/common/extract_info.py -f " + log_path + " -t caffe", shell=True))

time.sleep(10)
power, mem = cgp.get_average_gpu_power_and_mem(args.gpu, power_log_file)

if ".log" not in log_file:
        log_file += ".log"
log_path = os.getcwd() + "/" + log_file

#Save log file
#log_path="logs/" + log_file
with open(log_path, "a") as logFile:
	logFile.write("Total time: " + str(t) + "\n")
	logFile.write("Average power: " + str(power) + "\n")
	logFile.write("Average memory: " + str(mem) + "\n")
	logFile.write("cmd: " + cmd + "\n")
os.system("mv " + log_path + " logs")

print ('')
os.system("cat " + "logs/" + log_file)
