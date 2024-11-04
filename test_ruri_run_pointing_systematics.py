import numpy as np
import subprocess
import sys
import os

whoami = subprocess.run(['whoami'], capture_output=True, text=True)
jss_account = whoami.stdout.strip()
bizcode = os.getenv('bizcode') # load registered enviroment variable

# --------- JSS setting ----------- #
conda_base = f"/ssd/{jss_account[0]}/{jss_account}/.src/anaconda3/etc/profile.d/conda.sh"
conda_env = "lbs_wedge"
coderoot = f'/home/{jss_account[0]}/{jss_account}/data/program/e2e_sim/pointing_sys'
resource_unit = "RURI"
user_email = 'takase_y@s.okayama-u.ac.jp'  # your email for notification

vnode = 1 # 仮想ノード数, RURIは100が最大
vnode_core = 20 # ノードあたりの要求コア数, RURIは36が最大
vnode_mem = 64 # ノードあたりの要求メモリ量, RURIは5,850GBが最大

mode = "debug" # debugモードでの最大使用コア1800
#mode = "default"
job_name = "pntsys"
# When you use the `debug` mode you should requesgt <= 1800 == "00:30:00"
elapse = "02:30:00"
if mode == "debug":
    elapse = "00:15:00"

# --------- TOML file params setting ----------- #
# [general]
imo_path = f"/home/{jss_account[0]}/{jss_account}/data/litebird/litebird_imo/IMO/schema.json"
base_dir_name = "241104_test2_2hr"
imo_version = 'v2'
telescope = 'MFT'
nside_in = 128
nside_out = 128
cmb_seed = 33
cmb_r = 0.0
random_seed = 12345

# [simulation]
channel = 'M1-100'
det_names_file = 'detectors_'+telescope+'_'+channel+'_T+B'
base_path = os.path.join(coderoot, f'outputs/{base_dir_name}')
start_time = 0 # '2030-04-01T00:00:00' #float for circular motion of earth around Sun, string for ephemeridis
duration_s = 3600*2#*24*365 #simulated seconds
sampling_hz = 19.0
gamma = 0.0
wedge_angle_arcmin = 1.0
hwp_rpm = None # if None, the imo value will be used.
save_hitmap = False

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

# toml_filename   = str(uuid.uuid4())
toml_filename = 'pntsys_'+det_names_file+'_params'
tomlfile_path = os.path.join(ancillary, toml_filename+'.toml')
tomlfile_data = f"""
[hpc]
machine = '{resource_unit}'
vnode = {vnode}
vnode_mem = {vnode_mem}
vnode_core = {vnode_core}
elapse = '{elapse}'
mode = '{mode}'

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
save_hitmap = '{save_hitmap}'

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
#JX -L vnode={vnode}
#JX -L vnode-core={vnode_core}
#JX -L vnode-mem={vnode_mem}Gi

#JX -o {logdir}/%n_%j.out
#JX -e {logdir}/%n_%j.err
#JX --spath {logdir}/%n_%j.stats
#JX -N {job_name}
#JX -m e
#JX --mail-list {user_email}
#JX -S
export OMP_NUM_THREADS=1

module load intel
source {conda_base}
conda activate {conda_env}

cd {script_dir}
mpiexec -n {vnode_core} python -c "from e2e_pointing_systematics import pointing_systematics;
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
