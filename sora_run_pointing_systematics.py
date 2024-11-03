import numpy as np
import subprocess
import sys
import os
import uuid

# --------- JSS setting ----------- #
jss_account = "t541"
venv_base = "/ssd/t/t541/.src/lbsim_branches/lbs_hwp_wedge/bin/activate"
coderoot = '/home/t/t541/data/program/e2e_sim/pointing_sys'
resource_unit = "SORA"
bizcode = "DU10503"
user_email = 'takase_y@s.okayama-u.ac.jp'  # your email for notification

node = 4
node_mem = 28   # Unit: GiB, Upper limit=28GiB, Value when unspecified=28GiB
mpi_process = 48*node  # 48   # Upper limit of number of process per node is 48, it can be 48*`node`
mode = "debug"
#mode = "default"
#job_name = "dbg_pntsys"
job_name = "pntsys"
# When you use the `debug` mode you should requesgt <= 1800 == "00:30:00"
elapse = "01:00:00"
if mode == "debug":
    elapse = "00:30:00"

# --------- TOML file params setting ----------- #
# [general]
imo_path = "/home/t/t541/data/litebird/litebird_imo/IMO/schema.json"
base_dir_name = "hwp_wedge_1day"
imo_version = 'v2'
telescope = 'MFT'  # sys.argv[1] #e.g. 'LFT'
nside_in = 2048
nside_out = 512
cmb_seed = 33
cmb_r = 0.0
random_seed = 12345

# [simulation]
channel = 'M1-100'  # sys.argv[2] #e.g. 'L4-140'
det_names_file = 'detectors_'+telescope+'_'+channel+'_T+B'  # _case'+case]
base_path = os.path.join(coderoot, f'outputs/{base_dir_name}')
start_time = 0 # '2030-04-01T00:00:00' #float for circular motion of earth around Sun, string for ephemeridis
duration_s = 3600*24#*24*180#365 #simulated seconds
sampling_hz = 19.0
gamma = 0.0
wedge_angle_arcmin = 1.0
hwp_rpm = None # if None, the imo value will be used.

# --------- Setting is done, bottoms are automated ----------- #



script_dir = os.path.join(coderoot, 'scripts')
logdir = os.path.join(coderoot, 'log')
ancillary = os.path.join(coderoot, 'ancillary')
if not os.path.exists(base_path):
    os.makedirs(base_path, exist_ok=True)
if not os.path.exists(logdir):
    os.makedirs(logdir, exist_ok=True)
if not os.path.exists(ancillary):
    os.makedirs(ancillary, exist_ok=True)

#toml_uuid = str(uuid.uuid4())
toml_filename = 'pntsys_'+det_names_file+'_params'#+toml_uuid
toml_filename = 'pntsys_'+det_names_file+'_params'
tomlfile_path = os.path.join(ancillary, toml_filename+'.toml')
tomlfile_data = f"""
[hpc]
machine = '{resource_unit}'
node = {node}
node_mem = {node_mem}
mpi_process = {mpi_process}
elapse = '{elapse}'

[general]
imo_path = '{imo_path}'
imo_version = '{imo_version}'
telescope = '{telescope}'
det_names_file = '{det_names_file}'
nside_in = {nside_in}
nside_out = {nside_out}
random_seed = {random_seed}
cmb_seed = {cmb_seed}
cmb_r = {cmb_r}

[simulation]
base_path = '{base_path}'
gamma = {gamma}
start_time = {start_time}
duration_s = '{duration_s}'
sampling_hz = {sampling_hz}
wedge_angle_arcmin = {wedge_angle_arcmin}
hwp_rpm = '{hwp_rpm}'
"""
with open(tomlfile_path, 'w') as f:
    f.write(tomlfile_data)

jobscript_path = os.path.join(ancillary, det_names_file+".sh")
jobscript_data = f"""#!/bin/zsh
#JX --bizcode {bizcode}
#JX -L rscunit={resource_unit}
#JX -L rscgrp={mode}
#JX -L elapse={elapse}
#JX -L node={node}
#JX -L node-mem={node_mem}Gi
#JX --mpi proc={mpi_process}
#JX -o {logdir}/%n_%j.out
#JX -e {logdir}/%n_%j.err
#JX --spath {logdir}/%n_%j.stats
#JX -N {job_name}
#JX -m e
#JX --mail-list {user_email}
#JX -S
export OMP_NUM_THREADS=1

module purge
module load fjmpi-gcc/8.3.1
module load /opt/JX/modulefiles/aarch64/python/3.9.1
export LD_PRELOAD=/usr/lib/FJSVtcs/ple/lib64/libpmix.so

source {venv_base}
cd {script_dir}
mpiexec -n {mpi_process} python -c "from e2e_pointing_systematics import pointing_systematics;
pointing_systematics('{toml_filename}')"
"""

with open(jobscript_path, 'w') as f:
    f.write(jobscript_data)

process = subprocess.Popen("jxsub " + jobscript_path,
                           shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(stdout_data, stderr_data) = process.communicate()

# print useful information
print("out: "+str(stdout_data).split('b\'')[1][:-3])
print("err: "+str(stderr_data).split('b\'')[1][:-3])
print('')
