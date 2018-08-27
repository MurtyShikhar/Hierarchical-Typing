import subprocess
import itertools
import argparse
import sys
import os
from time import sleep
import random

parser = argparse.ArgumentParser()
parser.add_argument('-partition', default='titanx-short', type=str)
parser.add_argument('-cpu_memory', default='29GB', type=str)
parser.add_argument('-log_dir', default='logs', required=True, type=str)
parser.add_argument('-base_dir', default='', required=True, type=str)
parser.add_argument('-user', default='pat', required=True, type=str)
parser.add_argument('-dataset', default='typenet', type=str)
args = parser.parse_args(sys.argv[1:])

slurm_cmd = 'srun --gres=gpu:1 --partition=%s --mem=%s ' % (args.partition, args.cpu_memory)
base_cmd = 'python src/deploy.py -dataset %s -base_dir %s -epsilon 1e-4 -beta1 0.9 -beta2 0.9 -clip_val 10 -embedding_dim 300 -batch_size 1024 -save_model 0' % (args.dataset, args.base_dir)
log_dir = "%s" %args.log_dir
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

max_jobs = 300
dropout = [0.5, 0.75, .80, .85]
lr = [.001]
epsilon = [1e-4]
weight_decay = [1e-6, 3e-6, 5e-6, 1e-7]
encoder = ['position_cnn', 'basic']
struct_weight = [0.1, 0.2, 0.5, 1.0, 2.0]
asymmetric = [0, 1]
complex = [1]
all_params = [dropout, lr, weight_decay, encoder, struct_weight, asymmetric, complex]
names = ['dropout', 'lr', 'weight_decay', 'encoder', 'struct_weight', 'asymmetric', 'complex']

all_jobs = list(itertools.product(*all_params))
random.shuffle(all_jobs)
jobs_list = {}
for i, setting in enumerate(all_jobs):
    name_setting = {n: s for n, s in zip(names, setting)}
    # remove redundant settings
    #name_setting['bilinear_l2'] = name_setting['weight_decay']
    setting_list = ['-%s %s' % (name, str(value)) for name, value in name_setting.iteritems()]
    setting_str = ' '.join(setting_list)
    log_str = '_'.join(['%s-%s' % (n, str(s)) for n, s in name_setting.iteritems()])
    jobs_list[log_str] = setting_str

print('Running %d jobs and writing logs to %s' % (len(jobs_list), log_dir))
for log_str, setting_str in jobs_list.iteritems():
    full_cmd = '%s %s %s -model_name %s' % (slurm_cmd, base_cmd, setting_str, log_str)
    bash_cmd = '%s &> %s/%s &' % (full_cmd, log_dir, log_str)

    # only run max_jobs at once
    jobs = max_jobs
    while jobs >= max_jobs:
        jobs = int(subprocess.check_output('squeue  | grep %s | wc -l' % args.user, shell=True))
        sleep(1)
    print(bash_cmd)
    subprocess.call(bash_cmd, shell=True)

print('Done. Ran %d jobs.' % len(jobs_list))
