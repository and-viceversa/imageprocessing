import glob
import multiprocessing
import os
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from functools import partial
from pprint import pprint

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed
from mapboxgl.utils import df_to_geojson
from tqdm import tqdm

import micasense.imageutils as imageutils
import micasense.plotutils as plotutils
from micasense import utils
from micasense.capture import Capture
from micasense.imageset import ImageSet

warnings.filterwarnings('ignore')


def load_balancer(iterable, parameters):
    with ProcessPoolExecutor() as pool:
        # num cores to use in processing
        total_cores = multiprocessing.cpu_count()

        # counters to hold how many jobs are currently running on each
        mp_submitted = 0
        dask_submitted = 0

        # holds mp and dask futures
        mp_futures = []
        dask_futures = []

        # flag to determine if multiprocessing slots are open for the next image
        next_image = True

        # kwargs for tqdm
        kwargs = {
            'total': len(iterable),
            'unit': 'Capture',
            'unit_scale': False,
            'leave': True
        }

        # dask client
        client = Client(address=parameters['client'], direct_to_workers=True)

        # iterate over all captures in ImageSet
        for i in tqdm(iterable=iterable, desc='Processing ImageSet.captures ...', **kwargs):
            # if local and cluster are full, enter waiting loop
            if mp_submitted >= total_cores and dask_submitted >= total_cores:
                next_image = False

            # while False, subtract counts and save images. wait time to allow for other jobs to finish
            while next_image is False:
                # check if mp jobs are full. release done futures.
                for j, future in enumerate(mp_futures):
                    if future.done():
                        mp_submitted -= 1
                        del mp_futures[j]
                    else:
                        break

                # We break out of the waiting loop when multiprocessing is not full under the assumption
                # that preferring mp submits will always be faster than scanning the cluster
                if mp_submitted < total_cores:
                    next_image = True

                # check if dask jobs are full. release done futures.
                for e, future in enumerate(dask_futures):
                    if future.done():
                        dask_submitted -= 1
                        pool.submit(partial(save_capture, parameters), dask_futures.pop(e))
                    else:
                        break
                # condition for breaking out of waiting loop
                if mp_submitted < total_cores or dask_submitted < total_cores:
                    next_image = True

            # submit jobs to mp
            if mp_submitted < total_cores:
                mp_futures.append(pool.submit(partial(process_image_set, parameters), i))
                mp_submitted += 1
                continue
            # attempt to submit to dask. if img.raw() fails, send to local mp.
            elif dask_submitted < total_cores:
                try:
                    # load raw image into Capture before sending to cluster
                    [img.raw() for img in i.images]
                except Exception:
                    mp_futures.append(pool.submit(partial(process_image_set, parameters), i))
                    mp_submitted += 1
                    continue
                # scatter to cluster and start job
                big_future = client.scatter(i)
                dask_futures.append(client.submit(partial(dask_process_image_set, parameters), big_future))
                dask_submitted += 1
                continue
            else:
                print('FALLTHROUGH ERROR: {} WAS MISSED IN PROCESSING FIX YOUR CODE'.format(i))
                raise RuntimeError

        print('\tWaiting for cluster processes to finish...')

        # catch all remaining dask futures that may still be sitting on the cluster
        for future in dask_as_completed(dask_futures):
            pool.submit(partial(save_capture, parameters), future)


def parallel_process(function, iterable, parameters):
    """
    Multiprocessing Pool handler.
    :param function: function used in multiprocessing call
    :param iterable: iterable holding objects passed to function for each process
    :param parameters: dict of any function parameters other than the iterable object
    """

    # open a multiprocessing pool. no parameter value defaults to all cpu cores.
    with ProcessPoolExecutor() as pool:
        # run multiprocessing
        futures = [pool.submit(partial(function, parameters), i) for i in iterable]

        # kwargs for tqdm
        kwargs = {
            'total': len(futures),
            'unit': 'Capture',
            'unit_scale': False,
            'leave': True
        }

        # Receive Future objects as they complete. Print out the progress as tasks complete
        for _ in tqdm(iterable=as_completed(futures), desc='Processing ImageSet', **kwargs):
            pass


def process_image_set(params: dict, cap: Capture):
    """
    Process an ImageSet according to program parameters. Saves rgb
    :param params: dict of program parameters from main()
    :param cap: MicaSense Capture object
    """
    try:
        if len(cap.images) == params['capture_len']:
            cap.create_aligned_capture(irradiance_list=params['irradiance'], warp_matrices=params['warp_matrices'])
        else:
            print(f"Capture {cap.uuid} only has {len(cap.images)} Images. Should have {params['capture_len']}. "
                  f"Skipping...")
            return

        if params['save_stack']:
            output_stack_file_path = os.path.join(params['output_stack_dir'], cap.uuid + '.tif')
            if params['overwrite'] or not os.path.exists(output_stack_file_path):
                cap.save_capture_as_stack(output_stack_file_path)
        if params['save_rgb']:
            output_rgb_file_path = os.path.join(params['output_rgb_dir'], cap.uuid + '.jpg')
            if params['overwrite'] or not os.path.exists(output_rgb_file_path):
                cap.save_capture_as_rgb(output_rgb_file_path)

        cap.clear_image_data()
    except Exception as e:
        print(e)
        pprint(params)
        quit()


def dask_process_image_set(params: dict, cap: Capture):
    if len(cap.images) == params['capture_len']:
        cap.create_aligned_capture(irradiance_list=params['irradiance'], warp_matrices=params['warp_matrices'])
    else:
        print(f"Capture {cap.uuid} only has {len(cap.images)} Images. Should have {params['capture_len']}. "
              f"Skipping...")
        return None

    [img.clear_image_data() for img in cap.images]

    return cap


def save_capture(parameters, future):
    cap = future.result()
    if parameters['save_stack']:
        output_stack_file_path = os.path.join(parameters['output_stack_dir'], cap.uuid + '.tif')
        if parameters['overwrite'] or not os.path.exists(output_stack_file_path):
            cap.save_capture_as_stack(output_stack_file_path)
    if parameters['save_rgb']:
        output_rgb_file_path = os.path.join(parameters['output_rgb_dir'], cap.uuid + '.jpg')
        if parameters['overwrite'] or not os.path.exists(output_rgb_file_path):
            cap.save_capture_as_rgb(output_rgb_file_path)

    cap.clear_aligned_capture()


def make_dir_if_not_exist(dir_dict):
    """
    Make file directories if they don't exist based on boolean value in dir_dict k:v.
    :param dir_dict: dict of {str directory_name : boolean}
    """
    [os.mkdir(d) for d in dir_dict if dir_dict[d] and not os.path.exists(d)]


def drop_low_altitudes(df, image_set):
    print('Dropping low altitudes from ImageSet...')
    cutoff_altitude = df.altitude.mean() - (2.0 * df.altitude.std())
    print('\tCutoff altitude is {}'.format(cutoff_altitude))
    print('\tMean altitude is {}'.format(df.altitude.mean()))

    total_num = len(image_set.captures)

    for i, cap in enumerate(image_set.captures):
        if float(cap.images[0].altitude) < float(cutoff_altitude):
            image_set.captures[i] = None

    image_set.captures = [i for i in image_set.captures if i is not None]

    num_to_process = len(image_set.captures)

    print('\tThere were {} total captures. {} remain.'.format(total_num, num_to_process))


def drop_slow_shutter_speeds(image_set):
    print('Dropping slow shutter speeds (>= 0.01) from ImageSet...')

    total_num = len(image_set.captures)

    for i, cap in enumerate(image_set.captures):
        if cap.images[0].exposure_time >= 0.01:
            image_set.captures[i] = None

    image_set.captures = [i for i in image_set.captures if i is not None]

    num_to_process = len(image_set.captures)

    print('\tThere were {} total captures. {} remain.'.format(total_num, num_to_process))


def write_panel_diagnostics(panel_capture, output_diagnostic_dir, irradiance_list):
    print('Plotting diagnostic graphs...')
    # save panel raw plots
    panel_capture.plot_raw(show=False, file_path=os.path.join(output_diagnostic_dir, '1_panel_raw'))

    # save panel plots
    panel_capture.plot_panels(show=False, file_path=os.path.join(output_diagnostic_dir, '2_panel_plots'))

    # save panel radiance plots
    panel_capture.plot_radiance(show=False, file_path=os.path.join(output_diagnostic_dir, '3_panel_radiance'))

    # save panel undistorted radiance plots
    panel_capture.plot_undistorted_radiance(show=False,
                                            file_path=os.path.join(
                                                output_diagnostic_dir, '4_panel_undistorted_radiance')
                                            )

    # save panel vignette plots
    panel_capture.plot_vignette(show=False, file_path=os.path.join(output_diagnostic_dir, '5_panel_vignette'))

    # save panel undistorted reflectance plots
    panel_capture.plot_undistorted_reflectance(irradiance_list=irradiance_list,
                                               show=False,
                                               file_path=os.path.join(
                                                   output_diagnostic_dir, '6_panel_undistorted_reflectance')
                                               )

    # save panel region reflectance gaussian blur
    print('\tOrdered Panel Coordinates are as follows:')
    for i, panel in enumerate(panel_capture.panels):
        ul, ll, ur, lr = panel.ordered_panel_coordinates()
        print(f'\t\t{ul, ll, ur, lr}')

        reflection_image = panel_capture.images[i].reflectance(irradiance=irradiance_list[i])
        panel_region_reflectance = reflection_image[ul[1]:lr[1], ul[0]:lr[0]]
        panel_region_reflectance_blur = cv2.GaussianBlur(panel_region_reflectance, (55, 55), 5)

        plotutils.plot_with_color_bar(img=panel_region_reflectance_blur,
                                      title=f'Band {i} Image Panel Region Reflectance',
                                      plot_text='Min Reflectance in panel region: {:1.2f}\n'
                                                'Max Reflectance in panel region: {:1.2f}\n'
                                                'Mean Reflectance in panel region: {:1.2f}\n'
                                                'Standard deviation in region: {:1.4f}\n'
                                                'Ideal is <3% absolute reflectance'.format(
                                          panel_region_reflectance.min(),
                                          panel_region_reflectance.max(),
                                          panel_region_reflectance.mean(),
                                          panel_region_reflectance.std()),
                                      show=False,
                                      file_path=os.path.join(output_diagnostic_dir,
                                                             f'refl_panel_region_blur_band_{i}'))


def write_flight_diagnostics(columns, df, output_diagnostic_dir):
    print('Plotting flight diagnostics...')
    # plot capture metadata
    columns.remove('capture_id')
    df.drop(columns='capture_id')
    for i in range(3, len(columns) - 1):
        fig, ax = plt.subplots()
        df.plot(y=columns[i], ax=ax, subplots=True, figsize=(16, 10), style=['b'])
        ax.legend(loc='upper right', fancybox=True, shadow=True)
        fig.savefig(fname=os.path.join(output_diagnostic_dir, 'capture_overview_{}'.format(columns[i].lower())),
                    bbox_inches='tight')
        del fig, ax


def write_metadata(image_set, output_path, save_stack, save_rgb):
    """
    Extract Metadata from Captures list and save to log.csv.
    :param image_set: MicaSense ImageSet
    :param output_path: str system file path to top level dir
    :param save_stack: boolean If True, write metadata to stacks.
    :param save_rgb: boolean If True, write metadata to RGB output.
    :return: None
    """

    exiftool_cmd = 'exiftool'

    # metadata output header
    header = "SourceFile,\
    GPSDateStamp,GPSTimeStamp,\
    GPSLatitude,GpsLatitudeRef,\
    GPSLongitude,GPSLongitudeRef,\
    GPSAltitude,GPSAltitudeRef,\
    FocalLength,\
    XResolution,YResolution,ResolutionUnits\n"

    lines = [header]
    rgb_lines = [header]

    # ----- Write metadata to log.csv -----
    for capture in image_set.captures:
        # get lat, lon, alt, time
        output_filename = capture.uuid + '.tif'
        full_output_path = os.path.join(output_path, '..', '_stacks', output_filename)
        rgb_output_path = os.path.join(output_path, '..', '_rgb', output_filename.replace('.tif', '.jpg'))
        lat, lon, alt = capture.location()

        # write to csv in format:
        # IMG_0199_1.tif,"33 deg 32' 9.73"" N","111 deg 51' 1.41"" W",526 m Above Sea Level
        lat_deg, lat_min, lat_sec = utils.dd_2_dms(lat)
        lon_deg, lon_min, lon_sec = utils.dd_2_dms(lon)
        lat_dir = 'North'
        if lat_deg < 0:
            lat_deg = -lat_deg
            lat_dir = 'South'
        lon_dir = 'East'
        if lon_deg < 0:
            lon_deg = -lon_deg
            lon_dir = 'West'
        resolution = capture.images[0].focal_plane_resolution_px_per_mm

        line_str = '"{}",'.format(full_output_path)
        line_str += capture.utc_time().strftime("%Y:%m:%d,%H:%M:%S,")
        line_str += '"{:d} deg {:d}\' {:.2f}"" {}",{},'.format(int(lat_deg), int(lat_min), lat_sec, lat_dir[0], lat_dir)
        line_str += '"{:d} deg {:d}\' {:.2f}"" {}",{},{:.1f} m Above Sea Level,Above Sea Level,'.format(int(lon_deg),
                                                                                                        int(lon_min),
                                                                                                        lon_sec,
                                                                                                        lon_dir[0],
                                                                                                        lon_dir, alt)
        line_str += '{}'.format(capture.images[0].focal_length)
        line_str += '{},{},mm'.format(resolution, resolution)
        line_str += '\n'  # when writing in text mode, the write command will convert to os.linesep
        lines.append(line_str)

        line_str = line_str[line_str.find(','):]
        line_str = '"{}"'.format(rgb_output_path) + line_str
        rgb_lines.append(line_str)

    full_csv_path = os.path.join(output_path, '..', 'log.csv')
    rgb_csv_path = os.path.join(output_path, '..', 'rgb_log.csv')
    with open(full_csv_path, 'w') as csv_file:
        csv_file.writelines(lines)
    with open(rgb_csv_path, 'w') as csv_file:
        csv_file.writelines(rgb_lines)

    # ----- Write metadata to images ----
    with ProcessPoolExecutor() as pool:
        if save_stack:
            cmd = '{} -csv="{}" -overwrite_original "{}"'.format(exiftool_cmd,
                                                                 full_csv_path,
                                                                 os.path.join(output_path, '..', '_stacks')
                                                                 )
            print(cmd)
            subprocess.call(cmd, shell=True)
            # pool.submit(subprocess.call, cmd, 'shell=True')

        if save_rgb:
            if os.environ.get('exiftoolpath') is not None:
                exiftool_cmd = os.path.normpath(os.environ.get('exiftoolpath'))
            else:
                exiftool_cmd = 'exiftool'

            cmd = '{} -csv="{}" -overwrite_original "{}"'.format(exiftool_cmd,
                                                                 rgb_csv_path,
                                                                 os.path.join(output_path, '..', '_rgb')
                                                                 )
            print(cmd)
            subprocess.call(cmd, shell=True)
            # pool.submit(subprocess.call, cmd, 'shell=True')


def process_all(project_directory, flight_data_directories, panel_serial, panel_albedo_defaults,
                save_stack, save_thumbnails, save_diagnostics, use_dls, overwrite, use_dask,
                client_address, alignment_paths, panel_paths, drop_low_altitude, match_index,
                max_alignment_iterations, pyramid_levels):
    warp_mode = cv2.MOTION_HOMOGRAPHY  # MOTION_HOMOGRAPHY or MOTION_AFFINE.

    # verify some output is True
    if not any((save_stack, save_diagnostics, save_thumbnails)):
        raise RuntimeError('No output requested. Quitting.')

    # verify number of flights is correct for batch processing
    if not len(flight_data_directories) == len(alignment_paths) == len(panel_paths):
        raise RuntimeError('drone_config.yml parameters flight_data_directories, alignment_paths, '
                           'and panel_paths must have save length for batch processing.')

    # process each flight
    for i, d in enumerate(flight_data_directories):
        # make directories for requested output
        print(f'Setting up required directories in {d}...')
        output_stack_dir = os.path.join(d, '..', '_stacks')
        output_rgb_dir = os.path.join(d, '..', '_rgb')
        output_diagnostic_dir = os.path.join(d, '..', '_diagnostic_output')

        make_dir_if_not_exist({output_diagnostic_dir: save_diagnostics,
                               output_rgb_dir: save_thumbnails,
                               output_stack_dir: save_stack})

        # Images to derive warp matrices. Alignment images perform best with man made features.
        if alignment_paths[i]:
            alignment_images = glob.glob(alignment_paths[i])
        else:
            # use rig relatives
            alignment_images = None

        ##################################################
        # Get reflectance panel Capture and panel albedo #
        ##################################################

        # Images to derive panel information.
        if panel_paths[i]:
            panel_images = glob.glob(panel_paths[i])
        else:
            # process radiance only
            panel_images = None

        # make Capture object from the panel_images
        if panel_images is not None:
            print('Building panel Capture...')
            panel_capture = Capture.from_file_list(panel_images)
            print('Detected {} panels in panel Capture. This should be 5 for the MicaSense Altum.'
                  .format(panel_capture.detect_panels()))
        else:
            panel_capture = None

        # get panel albedo
        if panel_capture is not None:
            # case where reflectance regions are auto detected or automatically found
            if panel_capture.panel_albedo() is not None and not any(v is None for v in panel_capture.panel_albedo()):
                panel_albedo = panel_capture.panel_albedo()
                print('\tFound panel {} albedo values: {}'.format(panel_serial, panel_albedo))
            # case where at least 1 band albedo value cannot be found. use hardcoded values from panel.
            else:
                panel_albedo = panel_albedo_defaults
                print('\tCould not find panel albedo values from panel serial number.')
                print('\tUsing panel {} hardcoded values: {}'.format(panel_serial, panel_albedo))

            if not panel_capture.panels_in_all_expected_images():
                print('{}'.format([i + 1 for i, p in enumerate(panel_capture.panels) if p.panel_corners() is None]))
                raise ValueError('Panel reflectance region not detected in the above bands.')

            irradiance_list = panel_capture.panel_irradiance(panel_albedo) + [0]
            print('\tImage type output is: reflectance')
        else:
            if use_dls:
                print('\tNo panel Capture entered. use_dls is True. Attempting to use DLS for reflectance values '
                      'calibration. No diagnostic images will be created.')
                irradiance_list = None
            else:
                print('\tNo panel Capture entered. use_dls value is False. Image output type will be: radiance. '
                      'No diagnostic images will be created.')
                irradiance_list = None
        # -----------------------------------------------

        # ----- get alignment data -----
        print('Computing alignment information...')
        if alignment_images is not None:
            print('\tComputing Warp Matrices...')
            alignment_capture = Capture.from_file_list(alignment_images)

            # get warp_matrices
            warp_matrices, _ = imageutils.align_capture(alignment_capture, ref_index=match_index,
                                                        max_iterations=max_alignment_iterations,
                                                        warp_mode=warp_mode,
                                                        pyramid_levels=pyramid_levels)
        else:
            print('\tUsing Rig Relatives...')
            warp_matrices = None

        # ----- Get all Captures -----
        print(f'Building ImageSet from {d}')
        image_set = ImageSet.from_directory(d, use_tqdm=True)

        # ----- Get all capture info as pandas DataFrame and visualize flight stats -----
        print('Building pandas DataFrame...')
        data, columns = image_set.as_nested_lists()
        df = pd.DataFrame.from_records(data, index='timestamp', columns=columns)

        # ----- Save geojson data for later -----
        print(f'Saving GEOJSON data to {os.path.join(d, "..")}')
        geojson_data = df_to_geojson(df, columns[3:], lat='latitude', lon='longitude')
        with open(os.path.join(d, '..', 'imageSet.json'), 'w') as f:
            f.write(str(geojson_data))

        # plot diagnostics
        if save_diagnostics:
            write_panel_diagnostics(panel_capture=panel_capture, output_diagnostic_dir=output_diagnostic_dir,
                                    irradiance_list=irradiance_list)
            write_flight_diagnostics(columns=columns, df=df, output_diagnostic_dir=output_diagnostic_dir)

        # if only diagnostics were requested, quit.
        if not any((save_stack, save_thumbnails)):
            quit()

        # drop high and low altitudes
        if drop_low_altitude:
            drop_low_altitudes(df=df, image_set=image_set)

        # free up dataframe memory
        del data, columns, df, geojson_data

        # ----- parallel process ImageSet -----
        if not use_dask:
            parallel_process(function=process_image_set,
                             iterable=image_set.captures,
                             parameters={'irradiance': irradiance_list,
                                         'warp_matrices': warp_matrices,
                                         'capture_len': len(image_set.captures[0].images),
                                         'output_stack_dir': output_stack_dir,
                                         'output_rgb_dir': output_rgb_dir,
                                         'save_stack': save_stack,
                                         'save_rgb': save_thumbnails,
                                         'overwrite': overwrite}
                             )
        elif use_dask and client_address:
            load_balancer(
                iterable=image_set.captures,
                parameters={'irradiance': irradiance_list,
                            'warp_matrices': warp_matrices,
                            'capture_len': len(image_set.captures[0].images),
                            'output_stack_dir': output_stack_dir,
                            'output_rgb_dir': output_rgb_dir,
                            'save_stack': save_stack,
                            'save_rgb': save_thumbnails,
                            'overwrite': overwrite,
                            'client': client_address}
            )
        else:
            raise RuntimeError('Fix use_dask flag or use_dask .yml value. Quitting ...')

        # ----- Write metadata to log.csv and output images -----
        print('Writing log.csv and adding metadata to output stacks...')
        write_metadata(image_set, d, save_stack, save_thumbnails)
