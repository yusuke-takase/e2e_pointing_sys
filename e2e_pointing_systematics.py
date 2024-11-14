import litebird_sim as lbs
import numpy as np
import healpy as hp
import matplotlib.pylab as plt
from astropy.time import Time
#import time
import pathlib
import os
import sys
import toml
from litebird_sim import mpi
import json
from litebird_sim import TimeProfiler, profile_list_to_speedscope
from logging import getLogger
from distutils.util import strtobool

logger = getLogger(__name__)

def plot_map(m, title, save_filename):
    '''
    This function saves the mollview of the map m as a png figure and returns
    a tuple used to show the image in the report.

    m: map to be plotted;
    title: string, shown in the title of the figure;
    save_filename: string, name of the output file, e.g. 'my_figure.png'

    returns: tutle with figure and save_filename
    '''

    fig = plt.figure()
    hp.mollview(m, title=title, fig=fig)
    return (fig, save_filename)

def save_append_maps(map_path,map_name,map_output,cov_output,figures):
    '''
    This function saves a set of T,Q,U maps (.fits), its covariance (.npy) and
    updates the figures list with its mollweide projection.

    map_path: path where to save maps, i.e. base_path+'maps/';
    map_name: name of the map, i.e. case under study;
    map_output: map to be saved;
    cov_output: cov to be saved;
    figures: list with tuples of (fig, save_filename) that has to be updated

    returns: updated figures list
    '''

    #save maps
    hp.write_map(map_path+map_name+'.fits',
                 map_output,
                 overwrite=True)

    #save cov
    if(cov_output is not None):
        np.save(map_path+map_name+'_cov.npy',
                cov_output)

    #update figures list
    fields = ['T','Q','U']
    for i in range(3):
        figures.append(plot_map(map_output[i],
                                title=map_name+' - '+fields[i],
                                save_filename=map_name+'_'+fields[i]+'.png'))
    return figures


def get_simulation(toml_filename: str, comm):
    """Generate simulation class

    Args:
        toml_filename (str): name of toml file which includes simulation infomation.
        comm: MPI communicator
    """
    tomlfile_path = os.path.dirname(
        os.getcwd())+"/ancillary/"+toml_filename+".toml"
    # load toml file
    toml_data = toml.load(open(tomlfile_path))
    random_seed = int(toml_data["general"]["random_seed"])
    imo_path = toml_data["general"]["imo_path"]

    # initiarize IMO
    imo = lbs.Imo(flatfile_location=imo_path)

    # create simulation
    sim = lbs.Simulation(
        parameter_file=tomlfile_path,
        random_seed=random_seed,
        imo=imo,
        mpi_comm=comm
    )
    # store parameters from toml file
    imo_version = sim.parameters["general"]["imo_version"]
    telescope = sim.parameters["general"]["telescope"]
    det_names_file = sim.parameters["general"]["det_names_file"]
    sampling_hz = sim.parameters["simulation"]["sampling_hz"]
    hwp_rpm = sim.parameters["simulation"]["hwp_rpm"]
    if hwp_rpm == 'None':
        hwp_rpm = None # hwp_rpm is determined by IMO
    else:
        hwp_rpm = float(hwp_rpm) # hwp_rpm is determined by specified value by toml file

    # read channel, noise and detector names
    det_names_file_path = os.path.dirname(
        os.getcwd())+"/ancillary/detsfile/"+det_names_file+".txt"
    det_file = np.genfromtxt(det_names_file_path,
                            skip_header=1,
                            dtype=str)

    channels = det_file[:, 1]  # [det_file[1]]
    # [det_file[4].astype(dtype=float)]
    noises = det_file[:, 4].astype(dtype=float)
    detnames = [det_file[:, 5][0]]  # [det_file[5]]

    # number of detectors = raws of {det_names_file}.txt
    n_det = np.size(detnames)

    # loading the instrument metadata
    # load the definition of the instrument
    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            imo,
            f"/releases/{imo_version}/satellite/{telescope}/instrument_info",
        )
    )

    #sim.set_scanning_strategy(
    #    imo_url=f"/releases/{imo_version}/satellite/scanning_parameters/",
    #    delta_time_s=delta_time_s,
    #)

    if hwp_rpm is not None:
        sim.set_hwp(lbs.IdealHWP(hwp_rpm * 2 * np.pi / 60)) # set hwp_rpm used in simulation by specified value in toml file
        sim.instrument.hwp_rpm = hwp_rpm
    else:
        sim.set_hwp(lbs.IdealHWP(sim.instrument.hwp_rpm * 2 * np.pi / 60)) # load hwp_rpm from IMO

    # store detector info to list
    dets = []
    for i_det in range(n_det):
        det = lbs.DetectorInfo.from_imo(
            url="/releases/"+imo_version+"/satellite/"+telescope+"/" +
                channels[i_det]+"/"+detnames[i_det]+"/detector_info",
            imo=imo
        )
        det.sampling_rate_hz = sampling_hz
        dets.append(det)
    return sim, dets, imo, channels, detnames

def printlog(message, rank):
    if(rank==0):
        #print(message) # show message as a standerd output
        logger.info(message)

def pointing_systematics(toml_filename):
    comm = lbs.MPI_COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Hello world: rank {rank} of {size} process")
    comm.barrier()

    # measure computation time within the nest
    perf_name = "start" # name of the nest, this is recorded in profile.json
    with TimeProfiler(name=perf_name, my_param=perf_name) as perf:
        # generate simulation and store values from toml file
        sim, dets, imo, channels, detnames = get_simulation(toml_filename, comm)
        hpc_info = sim.parameters["hpc"]
        general_info = sim.parameters["general"]
        simlation_info = sim.parameters["simulation"]

        imo_version = sim.parameters["general"]["imo_version"]
        telescope = sim.parameters["general"]["telescope"]
        det_names_file = sim.parameters["general"]["det_names_file"]
        nside_in = int(sim.parameters["general"]["nside_in"])
        nside_out = int(sim.parameters["general"]["nside_out"])
        cmb_seed = int(sim.parameters["general"]["cmb_seed"])
        cmb_r = sim.parameters["general"]["cmb_r"]
        save_hitmap = bool(strtobool(sim.parameters["general"]["save_hitmap"]))

        base_path = sim.parameters["simulation"]["base_path"]
        start_time = int(sim.parameters["simulation"]["start_time"])
        duration_s = sim.parameters["simulation"]["duration_s"]
        sampling_hz = sim.parameters["simulation"]["sampling_hz"]
        delta_time_s = sim.parameters["simulation"]["delta_time_s"]
        gamma = sim.parameters["simulation"]["gamma"]
        wedge_angle_arcmin = sim.parameters["simulation"]["wedge_angle_arcmin"]

        ch_info = lbs.FreqChannelInfo.from_imo(
            url="/releases/"+imo_version+"/satellite/" +
                telescope+"/"+channels[0]+"/channel_info",
            imo=sim.imo)
        n_det = len(dets)

    # produce cmb and fg maps; rank 0 reads maps and broadcasts them to the other processors
    if(rank==0):
        perf_list = [] # make list for conputation time profile
        perf_list.append(perf) # result of `perf_name = "start"` is stored into perf_list
        printlog("======= Prepare input maps ========", rank)

        perf_name = "run_mbs"
        with TimeProfiler(name=perf_name, my_param=perf_name) as perf:
            Mbsparams = lbs.MbsParameters(
                cmb_r=cmb_r,
                make_cmb=True,
                make_fg=False,
                seed_cmb=cmb_seed,
                fg_models=["pysm_synch_0", "pysm_dust_0"],
                gaussian_smooth=True,
                bandpass_int=False,
                nside=nside_in,
                units="uK_CMB",
                maps_in_ecliptic=False, # maps saved in ecliptic, because of dipole
            )
            mbs = lbs.Mbs(
                simulation=sim,
                parameters=Mbsparams,
                channel_list=ch_info
            )
            maps = mbs.run_all()[0]
            printlog("======= Inputmap is generated ========", rank)
        perf_list.append(perf)
    else:
        maps = None

    comm.barrier()
    del sim # delete sim because it is not used any more
    maps = comm.bcast(maps, root=0)
    comm.barrier()

    for i_run, syst in enumerate([True, False]):
        # First, the loop runs to generate pointings w/ systematics.
        # Seccond, the loop runs to generate pointings w/o systematics. Then, generate TODs.
        printlog(
            f"======= start obs {i_run}-th loop ========",
            rank
        )

        perf_name = f"run_pointings_{i_run}th"
        with TimeProfiler(name=perf_name, my_param=perf_name) as perf:
            # generate simulation which is used in temporary.
            sim_temp, dets_temp, imo, _channels, _detnames = get_simulation(toml_filename, comm)
            printlog("======= sim_temp done ========", rank)
            comm.barrier()

            # create obserbvation
            (obs,) = sim_temp.create_observations(
                detectors=dets_temp,
                n_blocks_det=1,
                n_blocks_time=size,
                split_list_over_processes=False
            )

            scanning_strategy = lbs.SpinningScanningStrategy.from_imo(
                url= f"/releases/{imo_version}/satellite/scanning_parameters/",
                imo=imo,
            )

            printlog("======= create_observation done ========", rank)

            comm.barrier()

            sim_temp.spin2ecliptic_quats = scanning_strategy.generate_spin2ecl_quaternions(
                start_time=obs.start_time,
                time_span_s=obs.n_samples/obs.sampling_rate_hz,
                delta_time_s=delta_time_s,
            )
            nbyte = sim_temp.spin2ecliptic_quats.nbytes()
            printlog(f"*** Byte of `spin2ecliptic_quats` per rank: {nbyte/1e6} MB ***", rank)

            if syst == True:
                comm.barrier()
                pntsys = lbs.PointingSys(sim_temp, obs, dets_temp) # generate pointing sys. instance
                refractive_idx = 3.1 # set refractive index of HWP
                wedge_angle_rad = np.deg2rad(wedge_angle_arcmin / 60) # HWP wedge andle
                # set actual pointing shift andle by HWP wedge angle
                tilt_angle_rad = pntsys.hwp.get_wedgeHWP_pointing_shift_angle(
                    wedge_angle_rad,
                    refractive_idx
                )
                # rotation freq for circular pointing disturbance, now it is chosen from HWP freq.
                # so, it cause 1f synchronized systematic effect with HWP
                ang_speed_radpsec = sim_temp.instrument.hwp_rpm * 2 * np.pi / 60
                # initial phase of pointing shift position
                tilt_phase_rad = 0.0
                comm.barrier()
                # add disturbance to detector's quaternions given by IMO
                # now, `dets_temp` will be changed.
                pntsys.hwp.add_hwp_rot_disturb(tilt_angle_rad, ang_speed_radpsec, tilt_phase_rad)
                printlog("======= pntsys done ========", rank)
            comm.barrier()


            comm.barrier()
            printlog("======= create_observation done ========", rank)
            lbs.prepare_pointings(obs, sim_temp.instrument, sim_temp.spin2ecliptic_quats, hwp=sim_temp.hwp)

            printlog("======= prepare_pointings done ========", rank)

            if save_hitmap == True:
                comm.barrier()
                lbs.precompute_pointings(obs)
            comm.barrier()

            if syst == True:
                # if it is in systematics loop, we compute TODs by systematic pointing
                comm.barrier()

                # store TODs into sim_temp.observations
                lbs.scan_map_in_observations(
                    obs,
                    maps=maps,
                    input_map_in_galactic=True,
                    interpolation="linear",
                )
                printlog("======= scan_map done ========", rank)

                comm.barrier()
                del maps # input maps will not be needed, it is expected to allocate huge memory (nside=2048) so we delete it.

                # preserve TODs with systeamtics because `obs.tod` will be initialized.
                tods_sys = np.empty_like(obs.tod)
                tods_sys = obs.tod

                printlog("======= tods_sys copied ========", rank)

                if save_hitmap == True:
                    # save hitmap if it is save_hitmap==True
                    printlog("======= calc hitmap done ========", rank)

                    perf_name = "run_hitmap"
                    with TimeProfiler(name=perf_name, my_param=perf_name) as perf:
                        # calculate hitmap
                        npix_out = hp.nside2npix(nside_out)
                        hit_map = np.zeros(npix_out)
                        comm.barrier()
                        for i_det in range(n_det):
                            hitpix = hp.ang2pix(
                                nside_out,
                                obs.pointing_matrix[i_det,:,0], # theta
                                obs.pointing_matrix[i_det,:,1], # phi
                            )
                            # counts number of hit per pixel, we use `npix_out` as bins.
                            idet_hitmap, _ = np.histogram(hitpix, bins=np.arange(npix_out+1))
                            hit_map += idet_hitmap
                        comm.barrier()
                        # gather every hit_map per rank into same place
                        hit_map = comm.allreduce(hit_map, op=mpi.MPI.SUM)
                        comm.barrier()
                    if(rank==0):
                        perf_list.append(perf)
                        # save hitmap to fits file and append it to figure

                        # this dir. is used for saving maps
                        map_path = os.path.join(base_path, 'maps/')
                        if not os.path.exists(map_path):
                            os.mkdir(map_path)

                        # `figures` is called when we make a report, every figure should be appended into `figures`
                        figures = []
                        map_name = f'{telescope}_{channels[0]}_hitmap_{duration_s}s_wedge_{wedge_angle_arcmin}arcmin'
                        hp.write_map(map_path+map_name+'.fits', hit_map, overwrite=True)
                        figures.append(plot_map(hit_map, title=map_name, save_filename=map_name+'.png'))
                        printlog("======= hitmap saved ========", rank)
                    comm.barrier()

                    del hit_map # delete hit_map after save
                else:
                    # if save_hitmap == False
                    # we just prepare `figures`, `map_path` and directory to save output maps.
                    if(rank==0):
                        figures = []
                        map_path = os.path.join(base_path, 'maps/')
                        if not os.path.exists(map_path):
                            os.mkdir(map_path)
            else:
                # We forth to inject `tod_sys` into `sim_temp.observations`
                # after this, `sim_temp` has:
                #    - "TODs" calculated by systematic pointing
                #    - "pointings" calculated without pointing systematics
                # By using this `sim_temp` we can simply do map-making which estimates
                # observed maps with pointing systematics
                obs.tod = tods_sys
                # now we make an access `sim_temp` by `sim_syst` to make it is clear
                sim_syst = sim_temp
                obs_syst = obs
        if(rank==0):
            perf_list.append(perf)
            printlog(f"======= {i_run}-th loop done ========", rank)
        comm.barrier()

    # print how MPI worked
    descr = sim_syst.describe_mpi_distribution()
    print(descr)

    if(rank==0):
        # save 2 plots: time-TOD plot; freq-PS(power spectrum of TOD)
        # we can confirm the 1f systematic effect due to the HWP wedge by PS plot.
        save_filename = f"{telescope}_{channels[0]}_tod_ps_{duration_s}s_wedge_{wedge_angle_arcmin}arcmin.png"
        time_cutoff_s = 3600*3
        limit = int(time_cutoff_s*sampling_hz)
        i_det = 0
        time_array = obs_syst.get_times()[:limit]
        tods = obs_syst.tod[i_det][:limit]
        ps_syst = np.fft.fft(tods)
        freqs = np.fft.fftfreq(len(ps_syst), d=1.0/sampling_hz)

        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        ax[0].plot(
            time_array,
            tods,
            "-",
            label=f"w/ disturb. HWP wedge angle={wedge_angle_arcmin} arcmin.",
        )
        ax[0].set_xlim(0, 100)
        ax[0].set_xlabel("Time [s]")
        ax[0].set_ylabel(r"TODs [$\mu K$]")
        ax[0].legend()

        ax[1].plot(
            freqs,
            np.abs(ps_syst),
            ".",
            label=f"w/ disturb. HWP wedge angle={wedge_angle_arcmin} arcmin",
        )
        ax[1].axvline(
            sim_syst.instrument.hwp_rpm/60,
            linestyle="--",
            label="$1f$",
            color="black",
        )
        ax[1].axvline(
            4*sim_syst.instrument.hwp_rpm/60,
            linestyle="--",
            label="$4f$",
            color="black",
        )
        ax[1].set_yscale("log")
        ax[1].set_xlim(0, sampling_hz/2)
        ax[1].set_xlabel("Frequency [Hz]")
        ax[1].set_ylabel("Power Spectrum [$K/\sqrt{Hz}$]")
        ax[1].legend()
        figures.append((fig, save_filename))
        printlog("======= TOD and power spectrum plots saved ========", rank)

    printlog("======= Start binning map-making ========", rank)
    perf_name = "run_binner"
    with TimeProfiler(name=perf_name, my_param=perf_name) as perf:
        comm.barrier()
        # Do binning map-making
        # The TOD is expected to be generated higher nside_in than nside_out
        # in the case of production run. Here, the map will be down-graded by binner.
        # `obs_syst` has:
        #      TODs: simulated by systematic pointing
        #      obs_syst.pointing_matrix: w/o systeamtics
        # so we can just do map-making simply as bellow:
        binner_results = lbs.make_binned_map(
            nside=nside_out,
            observations=obs_syst,
            output_coordinate_system=lbs.CoordinateSystem.Galactic,
        )
    printlog("======= Binning map-making done ========", rank)

    if (rank == 0):
        perf_list.append(perf)
        printlog("======= Save output map ========", rank)
        # save the figure of output map.
        map_name = f'{telescope}_{channels[0]}_binnedmap_{duration_s}s_wedge_{wedge_angle_arcmin}arcmin'
        figures = save_append_maps(
            map_path,
            map_name,
            binner_results.binned_map,
            None, #cov_output,
            figures
        )
        sim_syst.append_to_report("""

## Run parameters

[HPC settings]

{% for item in hpc_info.items() %}
 - {{item[0]}} = {{item[1]}}
{% endfor %}

[General]

{% for item in general_info.items() %}
 - {{item[0]}} = {{item[1]}}
{% endfor %}

[Simulation]

{% for item in simulation_info.items() %}
 - {{item[0]}} = {{item[1]}}
{% endfor %}
- used_hwp_rpm = {{used_hwp_rpm}}
  - When the `hwp_rpm` is `None`, this number is used.

## Output maps

Produced output maps:

{% for figure in figs %}
 ![]({{ figure[1] }})
{% endfor %}

## Detector list

Detectors used in the simulation:

{% for detname in detnames %}
 `{{ detname }}`
{% endfor %}

## MPI info
{{descr}}

## How to read the output

### Maps and covariances

```python
import healpy
m = healpy.read_map("path/to/file.fits", field=[0, 1, 2])
```

### Covariances in NPY format

```python
import numpy as np
cov = np.load("path/to/filename.fits")
```

### TODs and pointings (observations)

```python
import litebird_sim as lbs

obs = lbs.io.read_one_observation("path/to/file.hdf5", limit_mpi_rank=False, tod_fields=['tod_name1','tod_name2', ...])
```

""",
            hpc_info = hpc_info,
            general_info=general_info,
            simulation_info=simlation_info,
            descr=descr,

            used_hwp_rpm=sim_syst.instrument.hwp_rpm,
            figures=figures,
            figs=figures, # needed to loop over figures
            detnames=detnames,
        )

        sim_syst.flush()

        message = "======= Simulation is finished ======="
        print(message)
        logger.info(message)

        with open(f"{base_path}/profile.json", "wt") as out_f:
            json.dump(profile_list_to_speedscope(perf_list), out_f)
