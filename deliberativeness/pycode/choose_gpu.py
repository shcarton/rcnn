import subprocess
import re
import os

gpus = [0,1,2,3]

# A line will look something like this: |    3      3395    C   python                                         547MiB |
pattern = '\| +([0-9]+) +([0-9]+) +C +\S+ +[0-9]+[a-zA-Z]+ +\|'


def check_gpu_usage(verbose=True):
	smi_str = subprocess.check_output(['nvidia-smi'])
	gpu_process_tuples = re.findall(pattern, smi_str)
	gpus_in_use = []
	if verbose:
		print 'Output of nvidia-smi:'
		print smi_str

		print 'GPUs process ownership details:'
		for gpu, process in gpu_process_tuples:
			try:
				ps_str = subprocess.check_output('ps -u -p {}'.format(process),shell=True)
				header, info,_ = ps_str.split('\n')
				print '\tGPU\t{}'.format(header)
				print '\t{}\t{}'.format(gpu, info)
				gpus_in_use.append(gpu)
			except Exception as ex:
				print ex.message()

	return gpus_in_use

def choose_gpu(verbose=True, default_gpu=3, force_num=None):
	'''
	Run nvidia-smi as a system command and choose an available GPU based on that
	:return:
	'''
	smi_str = subprocess.check_output(['nvidia-smi'])
	gpu_process_tuples = re.findall(pattern, smi_str)
	gpus_in_use = check_gpu_usage(verbose=verbose)

	if type(force_num) == int and force_num in gpus:
		print 'Using GPU {} regardless of what else is going on'.format(force_num)
		return force_num

	for gpu in gpus:
		if str(gpu) not in gpus_in_use:
			if verbose: print 'Selected GPU {} for use'.format(gpu)
			return gpu

	print("Could not find a free GPU. Enter a GPU number to try, or 'y' to use default GPU {}".format(default_gpu))
	response = raw_input()

	if response.lower().startswith('y'):
		print("Using default GPU {}".format(default_gpu))
		return default_gpu
	elif response.isdigit() and int(response) in gpus:
		print("Using selected GPU {}".format(response))
		return int(response)
	else:
		raise Exception('Could not find a free GPU. Check output of nvidia-smi command for details:\n{}'.format(smi_str))





def main():
	print 'Checking GPU usage'
	check_gpu_usage()


if __name__ == '__main__':
	main()