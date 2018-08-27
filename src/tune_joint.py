import subprocess
import itertools
import argparse
import sys
import os
from time import sleep
import random

parser = argparse.ArgumentParser()
parser.add_argument('-partition', default='titanx-short', type=str)
parser.add_argument('-cpu_memory', default='30GB', type=str)
parser.add_argument('-log_dir', default='logs', required=True, type=str)
parser.add_argument('-base_dir', default='', required=True, type=str)
parser.add_argument('-user', default='pat', required=True, type=str)
parser.add_argument('-dataset', default='typenet', type=str)
parser.add_argument('-take_frac',default=1.0, type=float)
args = parser.parse_args(sys.argv[1:])

slurm_cmd = 'srun --gres=gpu:1 --partition=%s --mem=%s ' % (args.partition, args.cpu_memory)
base_cmd = 'python src/deploy_mil.py -dataset typenet -base_dir %s -take_frac %5.4f -linker_weight 0.0 -clip_val 10 -num_epochs 5 -batch_size 10 -mode typing -embedding_dim 300 -lr 0.001 -beta1 0.9 -beta2 0.9 -epsilon 1e-4 -typing_weight 1 -test_batch_size 20' % (args.base_dir, args.take_frac)
log_dir = "%s_%5.4f" %(args.log_dir,args.take_frac)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

max_jobs = 125
dropout = [0.75, .80, .85]
weight_decay = [5e-6, 1e-5, 1e-7]
struct_weight = [0.0, 0.1, 0.5, 0.8, 1, 2.0, 4.0]
parent_sample_size = [16, 64, 256]
use_transitive = [0]
asymmetric = [0, 1]
complex = [0,1]
encoder = ['position_cnn', 'basic']
all_params = [dropout, weight_decay, struct_weight, parent_sample_size, use_transitive, asymmetric, encoder, complex]
names = ['dropout', 'weight_decay', 'struct_weight', 'parent_sample_size', 'use_transitive', 'asymmetric', 'encoder', 'complex']

all_jobs = list(itertools.product(*all_params))
random.shuffle(all_jobs)
jobs_list = {}
for i, setting in enumerate(all_jobs):
    name_setting = {n: s for n, s in zip(names, setting)}
    # remove redundant settings
    # remove redundant settings
    if name_setting['struct_weight'] == 0 and name_setting['parent_sample_size'] != 16: continue


    setting_list = ['-%s %s' % (name, str(value)) for name, value in name_setting.iteritems()]
    setting_str = ' '.join(setting_list)
    log_str = '_'.join(['%s-%s' % (n, str(s)) for n, s in name_setting.iteritems()])
    jobs_list[log_str] = setting_str

print('Running %d jobs and writing logs to %s' % (len(jobs_list), log_dir))
for log_str, setting_str in jobs_list.iteritems():
    full_cmd = '%s %s %s -model_name %s -save_model 0 ' % (slurm_cmd, base_cmd, setting_str, log_str)
    bash_cmd = '%s &> %s/%s &' % (full_cmd, log_dir, log_str)

    # only run max_jobs at once
    jobs = max_jobs
    while jobs >= max_jobs:
        jobs = int(subprocess.check_output('squeue  | grep %s | wc -l' % args.user, shell=True))
        sleep(1)
    print(bash_cmd)
    subprocess.call(bash_cmd, shell=True)

print('Done. Ran %d jobs.' % len(jobs_list))
