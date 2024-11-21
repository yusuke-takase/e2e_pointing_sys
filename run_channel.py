import subprocess

code_path = "/home/t/t541/data/program/e2e_sim/pointing_sys/scripts/sora_run_pointing_systematics.py"

channel_list = [
    'L1-040','L2-050','L1-060','L3-068','L2-068','L4-078','L1-078','L3-089','L2-089','L4-100','L3-119','L4-140',
    'M1-100','M2-119','M1-140','M2-166','M1-195',
    'H1-195','H2-235','H1-280','H2-337','H3-402'
]


wedge_angle_arcmin = 0.0
#channel = channel_list[1]

for channel in channel_list:
    process = subprocess.Popen(
        f"python {code_path} {channel} {wedge_angle_arcmin}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    (stdout_data, stderr_data) = process.communicate()

    # print useful information
    print("out: "+str(stdout_data).split('b\'')[1][:-3])
    print("err: "+str(stderr_data).split('b\'')[1][:-3])
    print('')
