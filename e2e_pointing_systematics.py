import litebird_sim as lbs
import numpy as np
import healpy as hp
import matplotlib.pylab as plt
from astropy.time import Time
#import time
import pathlib
import os
import sys
import copy
import toml
from litebird_sim import mpi
import json
from litebird_sim import TimeProfiler, profile_list_to_speedscope
from logging import getLogger
from distutils.util import strtobool

logger = getLogger(__name__)
#logger.info('message')

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
    toml_data = toml.load(open(tomlfile_path))
    random_seed = int(toml_data["general"]["random_seed"])
    imo_path = toml_data["general"]["imo_path"]

    imo = lbs.Imo(flatfile_location=imo_path)
    sim = lbs.Simulation(
        parameter_file=tomlfile_path,
        random_seed=random_seed,
        imo=imo,
        mpi_comm=comm
    )
    imo_version = sim.parameters["general"]["imo_version"]
    telescope = sim.parameters["general"]["telescope"]
    det_names_file = sim.parameters["general"]["det_names_file"]
    sampling_hz = sim.parameters["simulation"]["sampling_hz"]
    hwp_rpm = sim.parameters["simulation"]["hwp_rpm"]
    if hwp_rpm == 'None':
        hwp_rpm = None
    else:
        hwp_rpm = float(hwp_rpm)
    # read channel, noise and detector names
    det_names_file_path = os.path.dirname(
        os.getcwd())+"/ancillary/detsfile/"+det_names_file+".txt"
    det_file = np.genfromtxt(det_names_file_path,
                            skip_header=1,
                            dtype=str)

    channels = det_file[:, 1]  # [det_file[1]]
    # [det_file[4].astype(dtype=float)]
    noises = det_file[:, 4].astype(dtype=float)
    detnames = det_file[:, 5]  # [det_file[5]]

    # number of detectors = raws of {det_names_file}.txt
    ndet = np.size(detnames)

    # loading the instrument metadata
    # Load the definition of the instrument (MFT)
    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            imo,
            f"/releases/{imo_version}/satellite/{telescope}/instrument_info",
        )
    )
    sim.set_scanning_strategy(
        imo_url=f"/releases/{imo_version}/satellite/scanning_parameters/",
        delta_time_s=1.0 / sampling_hz,
    )
    if hwp_rpm is not None:
        sim.set_hwp(lbs.IdealHWP(hwp_rpm * 2 * np.pi / 60))
        sim.instrument.hwp_rpm = hwp_rpm
    else:
        sim.set_hwp(lbs.IdealHWP(sim.instrument.hwp_rpm * 2 * np.pi / 60))

    dets = []
    for i_det in range(ndet):
        det = lbs.DetectorInfo.from_imo(
            url="/releases/"+imo_version+"/satellite/"+telescope+"/" +
                channels[i_det]+"/"+detnames[i_det]+"/detector_info",
            imo=imo
        )
        det.sampling_rate_hz = sampling_hz
        dets.append(det)
    return sim, dets, channels, detnames

def pointing_systematics(toml_filename):
    comm = lbs.MPI_COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Hello world: rank {rank} of {size} process")

    perf_name = "start"
    with TimeProfiler(name=perf_name, my_param=perf_name) as perf:
        sim, dets, channels, detnames = get_simulation(toml_filename, comm)
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
        gamma = sim.parameters["simulation"]["gamma"]
        wedge_angle_arcmin = sim.parameters["simulation"]["wedge_angle_arcmin"]

        ch_info = lbs.FreqChannelInfo.from_imo(
            url="/releases/"+imo_version+"/satellite/" +
                telescope+"/"+channels[0]+"/channel_info",
            imo=sim.imo)
        ndet = len(dets)

    # produce cmb and fg maps; rank 0 reads maps and broadcasts them to the other processors
    if(rank==0):
        perf_list = []
        message = "======= Prepare input maps ========"
        print(message)
        logger.info(message)
        perf_name = "run_mbs"
        with TimeProfiler(name=perf_name, my_param=perf_name) as perf:
            Mbsparams = lbs.MbsParameters(
                cmb_r=cmb_r,
                make_cmb=True,
                make_fg=False,
                seed_cmb=cmb_seed,
                fg_models=["pysm_synch_0", "pysm_freefree_1", "pysm_dust_0"],
                gaussian_smooth=True,
                bandpass_int=False,
                nside=nside_in,
                units="uK_CMB",
                maps_in_ecliptic=False,  # maps saved in ecliptic, because of dipole
            )
            mbs = lbs.Mbs(
                simulation=sim,
                parameters=Mbsparams,
                channel_list=ch_info
            )
            maps = mbs.run_all()[0]
            message = "======= Inputmap is generated ========"
            print(message)
            logger.info(message)
        perf_list.append(perf)
    else:
        maps = None
    del sim

    maps = comm.bcast(maps, root=0)

    comm.barrier()

    for i_run, syst in enumerate([False, True]):
        # First, the loop runs to generate pointings w/o systematics.
        # Seccond, the loop runs to generate pointings w/o systematics. Then, generate TODs.
        if(rank==0):
            message = f"======= Create observations/pointings {i_run}-th loop ========"
            print(message)
            logger.info(message)

        perf_name = f"run_pointings_{i_run}th"
        with TimeProfiler(name=perf_name, my_param=perf_name) as perf:
            # make a tempolary simulation: `sim_temp` and detector list: `dets_temp`.
            # It has same values with `sim` that we declare first, though `sim_temp` and `dets_temp`
            # will be changed if the loop is syst==True, to impose systematics.
            sim_temp, dets_temp, _channels, _detnames = get_simulation(toml_filename, comm)
            if syst == True:
                # HWP wedge systematics will be imposed
                pntsys = lbs.PointingSys(sim_temp, dets_temp)

                # prepare parameters for systematics
                refractive_idx = 3.1
                wedge_angle_rad = np.deg2rad(wedge_angle_arcmin / 60)
                pntsys.hwp.tilt_angle_rad = pntsys.hwp.get_wedgeHWP_pointing_shift_angle(
                    wedge_angle_rad,
                    refractive_idx
                )
                pntsys.hwp.ang_speed_radpsec = sim_temp.instrument.hwp_rpm * 2 * np.pi / 60
                pntsys.hwp.tilt_phase_rad = 0.0
                comm.barrier()
                # Inject the systematics into the quaternions.
                # After this, `Detector.quats` will be changed.
                # Note that this wedge systeamtics happens in 19Hz. ]
                # So we have to prerare the quaternions in 19hz otherwise the
                # systemtics will not be expressed correctoly.
                pntsys.hwp.add_hwp_rot_disturb()

            comm.barrier()
            # create the ovserbations
            sim_temp.create_observations(
                detectors=dets_temp,
                n_blocks_det=1,
                n_blocks_time=size,
                split_list_over_processes=False
            )

            comm.barrier()
            # generate spin2eqliptic_quats
            sim_temp.prepare_pointings()

            comm.barrier()

            # generate pointings.
            # after this, the pointings can be accessed by
            # `sim_temp.observations[i_obs].pointing_matrix`.
            lbs.precompute_pointings(sim_temp.observations)

            comm.barrier()
            if(rank==0):
                # print the total memory alloc. by quats
                nbyte = sim_temp.spin2ecliptic_quats.nbytes()
                logger.info(f"Total byte of quats: {nbyte*size}")

            # Because we have already got pointings (θ,φ,ψ),
            # we don't need quaternions anymore. let's delete it to save memory.
            del sim_temp.spin2ecliptic_quats

            n_obs = len(sim_temp.observations)
            if syst == False:
                pointings_no_syst = []
                for i_obs in range(n_obs):
                    # store the pointings w/o systematics to use the "map-making"
                    pointings_no_syst.append(sim_temp.observations[i_obs].pointing_matrix)
                    if(rank==0):
                        nbyte = sys.getsizeof(pointings_no_syst)
                        logger.info(f"Total byte of pointing_matrix(w/o sys): {nbyte*size}")
                    # Since the pointings w/o systematics are have already stored in the list,
                    # we delete the .pointing_matrix to save memory.
                    del sim_temp.observations[i_obs].pointing_matrix
            else:
                pointings_syst = []
                for i_obs in range(n_obs):
                    # store the pointings w/ systematics to use "TOD-generation"
                    pointings_syst.append(sim_temp.observations[i_obs].pointing_matrix)
                    if(rank==0):
                        nbyte = sys.getsizeof(pointings_syst)
                        logger.info(f"Total byte of pointing_matrix(w/ sys): {nbyte*size}")
                    # Since the pointings w/ systematics are have already stored in the list,
                    # we delete the .pointing_matrix to save memory.
                    del sim_temp.observations[i_obs].pointing_matrix

                for i_obs in range(n_obs):
                    for i_det in range(ndet):
                        # this is the tricky part of this pipeline.
                        # the ψ with systematics is should not be affected by HWP wedge.
                        # i.e. is is expected to be same with ψ without systematics.
                        # However, due to the coding (maybe can be solved by sphistication)
                        # simulation has artifact.
                        # So we overwrite it by hands as below:
                        pointings_syst[i_obs][i_det,:,2] = pointings_no_syst[i_obs][i_det,:,2]

                comm.barrier()
                # make TODs by pointings w/ systematics
                # The TODs should be generated higher resolution (high nside) because
                # we have to simulate a TOD change due to the small pointing error.
                # In the production run, it is recommended to use nside_in=2048.
                lbs.scan_map_in_observations(
                    sim_temp.observations,
                    pointings=pointings_syst,
                    maps=maps,
                    input_map_in_galactic=True,
                    interpolation="linear",
                )
                # rename sim_temp to sim_syst because it is frozen.
                sim_syst = sim_temp
        if(rank==0):
            perf_list.append(perf)

    # print MPI infomation
    descr = sim_temp.describe_mpi_distribution()
    print(descr)

    if(rank==0):
        message = "======= Calculate hitmap ========"
        print(message)
        logger.info(message)

    if save_hitmap == True:
        # save a plot and fits file of hitmap which is calculated systemtics pointings
        perf_name = "run_hitmap"
        with TimeProfiler(name=perf_name, my_param=perf_name) as perf:
            npix_out = hp.nside2npix(nside_out)
            hit_map = np.zeros(npix_out)
            comm.barrier()
            for i_obs in range(n_obs):
                for i_det in range(ndet):
                    hitpix = hp.ang2pix(
                        nside_out,
                        pointings_syst[i_obs][i_det,:,0], # theta
                        pointings_syst[i_obs][i_det,:,1], # phi
                    )
                    iobs_idet_hitmap, _ = np.histogram(hitpix, bins=np.arange(npix_out+1))
                    hit_map += iobs_idet_hitmap
            comm.barrier()
            # after this, we don't need `pointing_sys` anymore, so delete it.
            del pointings_syst

            hit_map = comm.allreduce(hit_map, op=mpi.MPI.SUM)
            comm.barrier()
        if(rank==0):
            perf_list.append(perf)
            # save hitmap to fits file and append it to figure
            message = "======= Save hitmap ========"
            print(message)
            logger.info(message)
            map_path = os.path.join(base_path, 'maps/')
            if not os.path.exists(map_path):
                os.mkdir(map_path)

            figures = []
            map_name = f'{telescope}_{channels[0]}_hitmap_{duration_s}s_wedge_{wedge_angle_arcmin}arcmin'
            hp.write_map(map_path+map_name+'.fits', hit_map, overwrite=True)
            figures.append(plot_map(hit_map, title=map_name, save_filename=map_name+'.png'))
    else:
        # if save_hitmap == False
        # we just prepare `figures`, `map_path` and directory to save output maps.
        if(rank==0):
            figures = []
            map_path = os.path.join(base_path, 'maps/')
            if not os.path.exists(map_path):
                os.mkdir(map_path)
    del pointings_syst

    comm.barrier()

    if(rank==0):
        # save 2 plots: time-TOD plot; freq-PS(power spectrum of TOD)
        # we can confirm the 1f systematic effect due to the HWP wedge by PS plot.
        message = "======= Save TOD and power spectrum plots ========"
        print(message)
        logger.info(message)

        save_filename = f"{telescope}_{channels[0]}_tod_ps_{duration_s}s_wedge_{wedge_angle_arcmin}arcmin.png"
        time_cutoff_s = 3600*3
        limit = int(time_cutoff_s*sampling_hz)
        time_array = sim_syst.observations[0].get_times()[:limit]
        tods = sim_syst.observations[0].tod[0][:limit]
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

    if(rank==0):
        message = "======= Binning map-making ========"
        print(message)
        logger.info(message)

    perf_name = "run_binner"
    with TimeProfiler(name=perf_name, my_param=perf_name) as perf:
        comm.barrier()
        # Do binning map-making
        # The TOD is expected to be generated higher nside_in than nside_out
        # in the case of production run. Here, the map will be down-graded by binner.
        binner_results = lbs.make_binned_map(
            nside=nside_out,  # one can set also a different resolution than the input map
            observations=sim_syst.observations,
            pointings=pointings_no_syst,
            output_coordinate_system=lbs.CoordinateSystem.Galactic,
        )

    if (rank == 0):
        perf_list.append(perf)
        message = "======= Save output map ========"
        print(message)
        logger.info(message)
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
