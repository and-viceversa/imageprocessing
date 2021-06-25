#!/usr/bin/env python
# coding: utf-8
"""
MicaSense ImageSet Class

    An ImageSet contains a group of Captures. The Captures can be loaded from Image objects, from a list of files,
    or by recursively searching a directory for images.

Copyright 2017 MicaSense, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import fnmatch
import multiprocessing
import os

import exiftool

import micasense.capture as capture
import micasense.image as image
from micasense.imageutils import save_capture as save_capture


# FIXME: mirrors Capture.append_file(). Does this still belong here?
def image_from_file(filename):
    return image.Image(filename)


class ImageSet(object):
    """An ImageSet is a container for a group of Captures that are processed together."""

    def __init__(self, captures):
        self.captures = captures
        captures.sort()

    @classmethod
    def from_directory(cls, directory, progress_callback=None, use_tqdm=False, exiftool_path=None):
        """
        Create an ImageSet recursively from the files in a directory.
        :param directory: str system file path
        :param progress_callback: function to report progress to
        :param use_tqdm: boolean True to use tqdm progress bar
        :param exiftool_path: str system file path to exiftool location
        :return: ImageSet instance
        """
        cls.basedir = directory
        matches = []
        for root, dirnames, filenames in os.walk(directory):
            [matches.append(os.path.join(root, filename)) for filename in fnmatch.filter(filenames, '*.tif')]

        images = []

        if use_tqdm:  # to use tqdm progress bar instead of progress_callback
            from tqdm import tqdm
            with exiftool.ExifTool(exiftool_path) as exift:
                kwargs = {
                    'total': len(matches),
                    'unit': ' Files',
                    'unit_scale': False,
                    'leave': True
                }
                for i, path in tqdm(iterable=enumerate(matches), desc='Loading ImageSet', **kwargs):
                    images.append(image.Image(path, exiftool_obj=exift))
        else:
            with exiftool.ExifTool(exiftool_path) as exift:
                for i, path in enumerate(matches):
                    images.append(image.Image(path, exiftool_obj=exift))
                    if progress_callback is not None:
                        progress_callback(float(i) / float(len(matches)))

        # create a dictionary to index the images so we can sort them into captures
        # {
        #     "capture_id": [img1, img2, ...]
        # }
        captures_index = {}
        for img in images:
            c = captures_index.get(img.capture_id)
            if c is not None:
                c.append(img)
            else:
                captures_index[img.capture_id] = [img]
        captures = []
        for cap_imgs in captures_index:
            imgs = captures_index[cap_imgs]
            newcap = capture.Capture(imgs)
            captures.append(newcap)
        if progress_callback is not None:
            progress_callback(1.0)
        return cls(captures)

    def as_nested_lists(self):
        """
        Get timestamp, latitude, longitude, altitude, capture_id, dls-yaw, dls-pitch, dls-roll, irradiance from all
        Captures.
        :return: List data from all Captures, List column headers.
        """
        columns = [
            'timestamp',
            'latitude', 'longitude', 'altitude',
            'capture_id',
            'dls-yaw', 'dls-pitch', 'dls-roll'
        ]
        irr = ["irr-{}".format(wve) for wve in self.captures[0].center_wavelengths()]
        columns += irr
        data = []
        for cap in self.captures:
            dat = cap.utc_time()
            loc = list(cap.location())
            uuid = cap.uuid
            dls_pose = list(cap.dls_pose())
            irr = cap.dls_irradiance()
            row = [dat] + loc + [uuid] + dls_pose + irr
            data.append(row)
        return data, columns

    def dls_irradiance(self):
        """
        Get dict {utc_time : [irradiance, ...]} for each Capture in ImageSet.
        :return: None  # FIXME: This method appears to have no effect? Add return value?
        """
        series = {}
        for cap in self.captures:
            dat = cap.utc_time().isoformat()
            irr = cap.dls_irradiance()
            series[dat] = irr

    def save_stacks(self, warp_matrices, stack_directory, thumbnail_directory=None, irradiance=None, multiprocess=True,
                    overwrite=False, progress_callback=None):
        """
        Write band stacks and thumbnails to disk.
        :param warp_matrices: 2d List of warp matrices derived from Capture.get_warp_matrices()
        :param stack_directory: str system file path to output stack directory
        :param thumbnail_directory: str system file path to output thumbnail directory
        :param irradiance: List returned from Capture.dls_irradiance() or Capture.panel_irradiance()
        :param multiprocess: boolean True to use multiprocessing module
        :param overwrite: boolean True to overwrite existing files
        :param progress_callback: function to report progress to
        """

        if not os.path.exists(stack_directory):
            os.makedirs(stack_directory)
        if thumbnail_directory is not None and not os.path.exists(thumbnail_directory):
            os.makedirs(thumbnail_directory)

        save_params_list = []
        for cap in self.captures:
            save_params_list.append({
                'output_path': stack_directory,
                'thumbnail_path': thumbnail_directory,
                'file_list': [img.path for img in cap.images],
                'warp_matrices': warp_matrices,
                'irradiance_list': irradiance,
                'photometric': 'MINISBLACK',
                'overwrite_existing': overwrite,
            })

        if multiprocess:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            for i, _ in enumerate(pool.imap_unordered(save_capture, save_params_list)):
                if progress_callback is not None:
                    progress_callback(float(i) / float(len(save_params_list)))
            pool.close()
            pool.join()
        else:
            for params in save_params_list:
                # FIXME: save_capture() seems to duplicate Capture instances that already exist in an ImageSet.
                #  save_capture has no call to Capture.clear_image_data(). So while the duplicate Capture does fall
                #  out of scope, the original ImageSet isn't clearing any Captures that have already been saved.
                #  Recommend refactor to give ImageSet its own save logic, because it's accepting its own parameters
                #  from save_stack() anyway. imageutils.save_capture() could be preserved for a more general case.
                save_capture(params)
