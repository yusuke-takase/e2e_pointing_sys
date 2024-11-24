import subprocess

code_path = "/home/t/t541/data/program/e2e_sim/pointing_sys/scripts/single_sora_run_pointing_systematics.py"

nsides = [512, 1024, 2048, 4096]
wedge_angle_arcmin = [0.0, 1.0]
only_Ts = [True, False]
only_Ps = [True, False]
#channel = channel_list[1]

for nside in nsides:
    for wedge in wedge_angle_arcmin:
        for only_T in only_Ts:
            for only_P in only_Ps:
                if only_T == True and only_P == True:
                    pass
                else:
                    cmd = f"python {code_path} {nside} {wedge} {only_T} {only_P}"
                    print(cmd)
                    process = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                    (stdout_data, stderr_data) = process.communicate()
                    # print useful information
                    print("out: "+str(stdout_data).split('b\'')[1][:-3])
                    print("err: "+str(stderr_data).split('b\'')[1][:-3])
                    print('')
