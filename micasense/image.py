#!/usr/bin/env python
# coding: utf-8
"""
RedEdge Image Class

    An Image is a single file taken by a RedEdge or Altum camera representing one band of multispectral information.

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

import math
import os

import cv2
import exiftool
import numpy as np

import micasense.dls as dls
import micasense.metadata as metadata
import micasense.plotutils as plotutils


def rotations_degrees_to_rotation_matrix(rotation_degrees):
    """
    Helper function to convert euler angles to a rotation matrix.
    :param rotation_degrees: Tuple RigRelatives from metadata
    :return:
    """

    cx = np.cos(np.deg2rad(rotation_degrees[0]))
    cy = np.cos(np.deg2rad(rotation_degrees[1]))
    cz = np.cos(np.deg2rad(rotation_degrees[2]))
    sx = np.sin(np.deg2rad(rotation_degrees[0]))
    sy = np.sin(np.deg2rad(rotation_degrees[1]))
    sz = np.sin(np.deg2rad(rotation_degrees[2]))

    rx = np.mat([1, 0, 0,
                 0, cx, -sx,
                 0, sx, cx]).reshape(3, 3)
    ry = np.mat([cy, 0, sy,
                 0, 1, 0,
                 -sy, 0, cy]).reshape(3, 3)
    rz = np.mat([cz, -sz, 0,
                 sz, cz, 0,
                 0, 0, 1]).reshape(3, 3)
    return rx * ry * rz


class Image(object):
    """An Image is a single file taken by a RedEdge camera representing one band of multispectral information."""

    def __init__(self, image_path, exiftool_obj=None):
        if not os.path.isfile(image_path):
            raise IOError("Provided path is not a file: {}".format(image_path))
        self.path = image_path

        if exiftool_obj is not None:
            self.meta = metadata.Metadata(self.path, exiftool_obj=exiftool_obj)
        else:
            with exiftool.ExifTool() as exift:
                self.meta = metadata.Metadata(self.path, exiftool_obj=exift)

        if self.meta.band_name() is None:
            raise ValueError("Provided file path does not have a band name: {}".format(image_path))
        if self.meta.band_name().upper() != 'LWIR' and not self.meta.supports_radiometric_calibration():
            raise ValueError('Library requires images taken with RedEdge-(3/M/MX) camera firmware v2.1.0 or later. '
                             'Upgrade your camera firmware to at least version 2.1.0 to use this library with '
                             'RedEdge-(3/M/MX) cameras.')

        self.utc_time = self.meta.utc_time()
        self.latitude, self.longitude, self.altitude = self.meta.position()
        self.location = (self.latitude, self.longitude, self.altitude)
        self.dls_present = self.meta.dls_present()
        self.dls_yaw, self.dls_pitch, self.dls_roll = self.meta.dls_pose()
        self.xy_accuracy, self.z_accuracy = self.meta.get_gps_accuracy()
        self.capture_id = self.meta.capture_id()
        self.flight_id = self.meta.flight_id()
        self.band_name = self.meta.band_name()
        self.band_index = self.meta.band_index()
        self.black_level = self.meta.black_level()
        if self.meta.supports_radiometric_calibration():
            self.radiometric_cal = self.meta.radiometric_cal()
        self.exposure_time = self.meta.exposure()
        self.gain = self.meta.gain()
        self.bits_per_pixel = self.meta.bits_per_pixel()

        self.vignette_center = self.meta.vignette_center()
        self.vignette_polynomial = self.meta.vignette_polynomial()
        self.distortion_parameters = self.meta.distortion_parameters()
        self.principal_point = self.meta.principal_point()
        self.focal_plane_resolution_px_per_mm = self.meta.focal_plane_resolution_px_per_mm()
        self.focal_length = self.meta.focal_length_mm()
        self.focal_length_35 = self.meta.focal_length_35_mm_eq()
        self.center_wavelength = self.meta.center_wavelength()
        self.bandwidth = self.meta.bandwidth()
        self.rig_relatives = self.meta.rig_relatives()
        self.spectral_irradiance = self.meta.spectral_irradiance()

        self.auto_calibration_image = self.meta.auto_calibration_image()
        self.panel_albedo = self.meta.panel_albedo()
        self.panel_region = self.meta.panel_region()
        self.panel_serial = self.meta.panel_serial()

        self.rig_translations = None

        if self.dls_present:
            self.dls_orientation_vector = np.array([0, 0, -1])
            self.sun_vector_ned, \
            self.sensor_vector_ned, \
            self.sun_sensor_angle, \
            self.solar_elevation, \
            self.solar_azimuth = dls.compute_sun_angle(self.location,
                                                       self.meta.dls_pose(),
                                                       self.utc_time,
                                                       self.dls_orientation_vector)
            self.angular_correction = dls.fresnel(self.sun_sensor_angle)

            # when we have good horizontal irradiance the camera provides the solar az and el also
            if self.meta.scattered_irradiance() != 0 and self.meta.direct_irradiance() != 0:
                self.solar_azimuth = self.meta.solar_azimuth()
                self.solar_elevation = self.meta.solar_elevation()
                self.scattered_irradiance = self.meta.scattered_irradiance()
                self.direct_irradiance = self.meta.direct_irradiance()
                self.direct_to_diffuse_ratio = self.meta.direct_irradiance() / self.meta.scattered_irradiance()
                self.estimated_direct_vector = self.meta.estimated_direct_vector()
                if self.meta.horizontal_irradiance_valid():
                    self.horizontal_irradiance = self.meta.horizontal_irradiance()
                else:
                    self.horizontal_irradiance = self.compute_horizontal_irradiance_dls2()
            else:
                self.direct_to_diffuse_ratio = 6.0  # assumption
                self.horizontal_irradiance = self.compute_horizontal_irradiance_dls1()

            self.spectral_irradiance = self.meta.spectral_irradiance()
        else:  # no dls present or LWIR band: compute what we can, set the rest to 0
            self.dls_orientation_vector = np.array([0, 0, -1])
            self.sun_vector_ned, \
            self.sensor_vector_ned, \
            self.sun_sensor_angle, \
            self.solar_elevation, \
            self.solar_azimuth = dls.compute_sun_angle(self.location,
                                                       (0, 0, 0),
                                                       self.utc_time,
                                                       self.dls_orientation_vector)
            self.angular_correction = dls.fresnel(self.sun_sensor_angle)
            self.horizontal_irradiance = 0
            self.scattered_irradiance = 0
            self.direct_irradiance = 0
            self.direct_to_diffuse_ratio = 0

        # Internal image containers; these can use a lot of memory, clear with Image.clear_images
        self.__raw_image = None  # pure raw pixels
        self.__intensity_image = None  # black level and gain-exposure/radiometric compensated
        self.__radiance_image = None  # calibrated to radiance
        self.__reflectance_image = None  # calibrated to reflectance (0-1)
        self.__reflectance_irradiance = None
        self.__undistorted_source = None  # can be any of raw, intensity, radiance
        self.__undistorted_image = None  # current undistorted image, depending on source

    # solar elevation is defined as the angle between the horizon and the sun, so it is 0 when the
    # sun is at the horizon and pi/2 when the sun is directly overhead
    def horizontal_irradiance_from_direct_scattered(self):
        return self.direct_irradiance * np.sin(self.solar_elevation) + self.scattered_irradiance

    def compute_horizontal_irradiance_dls1(self):
        """This method is called from Image.__init__() if necessary. It does not need to be called during processing."""
        percent_diffuse = 1.0 / self.direct_to_diffuse_ratio
        # percent_diffuse = 5e4/(img.center_wavelength**2)
        sensor_irradiance = self.spectral_irradiance / self.angular_correction
        # find direct irradiance in the plane normal to the sun
        untilted_direct_irr = sensor_irradiance / (percent_diffuse + np.cos(self.sun_sensor_angle))
        self.direct_irradiance = untilted_direct_irr
        self.scattered_irradiance = untilted_direct_irr * percent_diffuse
        # compute irradiance on the ground using the solar altitude angle
        return self.horizontal_irradiance_from_direct_scattered()

    def compute_horizontal_irradiance_dls2(self):
        """
        Compute the proper solar elevation, solar azimuth, and horizontal irradiance for cases where the camera
        system did not do it correctly. This method is called from Image.__init__() if necessary. It does
        not need to be called during processing.
        """
        _, _, _, self.solar_elevation, self.solar_azimuth = dls.compute_sun_angle(self.location, (0, 0, 0),
                                                                                  self.utc_time, np.array([0, 0, -1]))
        return self.horizontal_irradiance_from_direct_scattered()

    def __lt__(self, other):
        return self.band_index < other.band_index

    def __gt__(self, other):
        return self.band_index > other.band_index

    def __eq__(self, other):
        return (self.band_index == other.band_index) and (self.capture_id == other.capture_id)

    def __ne__(self, other):
        return (self.band_index != other.band_index) or (self.capture_id != other.capture_id)

    def raw(self):
        """Lazy load the raw image once necessary."""
        if self.__raw_image is None:
            try:
                import rawpy
                self.__raw_image = rawpy.imread(self.path).raw_image
            except ImportError:
                self.__raw_image = cv2.imread(self.path, -1)
            except IOError:
                print("Could not open image at path {}".format(self.path))
                raise
            except Exception:
                self.__raw_image = cv2.imread(self.path, -1)
        return self.__raw_image

    def set_external_rig_relatives(self, external_rig_relatives):
        """
        Set external rig relatives.
        :param external_rig_relatives: TODO: Parameter docstring
        """
        self.rig_translations = external_rig_relatives['rig_translations']
        # external rig relatives are in rad
        self.rig_relatives = [np.rad2deg(a) for a in external_rig_relatives['rig_relatives']]
        px, py = external_rig_relatives['cx'], external_rig_relatives['cy']
        fx, fy = external_rig_relatives['fx'], external_rig_relatives['fy']
        rx = self.focal_plane_resolution_px_per_mm[0]
        ry = self.focal_plane_resolution_px_per_mm[1]
        self.principal_point = [px / rx, py / ry]
        self.focal_length = (fx + fy) * .5 / rx
        # TODO: set the distortion etc.

    def clear_image_data(self):
        """Clear all computed images to reduce memory overhead."""
        self.__raw_image = None
        self.__intensity_image = None
        self.__radiance_image = None
        self.__reflectance_image = None
        self.__reflectance_irradiance = None
        self.__undistorted_source = None
        self.__undistorted_image = None

    def size(self):
        """
        Get image size from metadata.
        :return: tuple int image width and int image height
        """
        width, height = self.meta.image_size()
        return width, height

    def reflectance(self, irradiance=None, force_recompute=False):
        """Lazy-compute and return a reflectance image provided an irradiance reference."""
        if self.__reflectance_image is not None and force_recompute is False and \
                (self.__reflectance_irradiance == irradiance or irradiance is None):
            return self.__reflectance_image
        if irradiance is None and self.band_name != 'LWIR':
            if self.horizontal_irradiance != 0.0:
                irradiance = self.horizontal_irradiance
            else:
                raise RuntimeError("Provide a band-specific spectral irradiance to compute reflectance.")
        if self.band_name != 'LWIR':
            self.__reflectance_irradiance = irradiance
            self.__reflectance_image = self.radiance() * math.pi / irradiance
        else:
            self.__reflectance_image = self.radiance()
        return self.__reflectance_image

    def intensity(self, force_recompute=False):
        """Lazy-computes and returns the intensity image after black level, vignette, and row correction applied.
        Intensity is in units of DN*Seconds without a radiance correction."""
        if self.__intensity_image is not None and force_recompute is False:
            return self.__intensity_image

        # get image dimensions
        image_raw = np.copy(self.raw()).T

        #  get radiometric calibration factors
        _, a2, a3 = self.radiometric_cal[0], self.radiometric_cal[1], self.radiometric_cal[2]

        # apply image correction methods to raw image
        v, x, y = self.vignette()
        r = 1.0 / (1.0 + a2 * y / self.exposure_time - a3 * y)
        level = v * r * (image_raw - self.black_level)
        level[level < 0] = 0
        max_raw_dn = float(2 ** self.bits_per_pixel)
        intensity_image = level.astype(float) / (self.gain * self.exposure_time * max_raw_dn)

        self.__intensity_image = intensity_image.T
        return self.__intensity_image

    def radiance(self, force_recompute=False):
        """Lazy-computes and returns the radiance image after all radiometric corrections have been applied."""
        if self.__radiance_image is not None and force_recompute is False:
            return self.__radiance_image

        # get image dimensions
        image_raw = np.copy(self.raw()).T

        if self.band_name != 'LWIR':
            #  get radiometric calibration factors
            a1, a2, a3 = self.radiometric_cal[0], self.radiometric_cal[1], self.radiometric_cal[2]
            # apply image correction methods to raw image
            v, x, y = self.vignette()
            r = 1.0 / (1.0 + a2 * y / self.exposure_time - a3 * y)
            level = v * r * (image_raw - self.black_level)
            level[level < 0] = 0
            max_raw_dn = float(2 ** self.bits_per_pixel)
            radiance_image = level.astype(float) / (self.gain * self.exposure_time) * a1 / max_raw_dn
        else:
            level = image_raw - (273.15 * 100.0)  # convert to C from K
            radiance_image = level.astype(float) * 0.01
        self.__radiance_image = radiance_image.T
        return self.__radiance_image

    def vignette(self):
        """Get a numpy array which defines the value to multiply each pixel by to correct for optical vignetting
        effects. Note: this array is transposed from normal image orientation and comes as part
        of a three-tuple, the other parts of which are also used by the radiance method.
        """

        # get vignette center
        vignette_center_x, vignette_center_y = self.vignette_center

        # get a copy of the vignette polynomial because we want to modify it here
        v_poly_list = list(self.vignette_polynomial)

        # reverse list and append 1., so that we can call with numpy polyval
        v_poly_list.reverse()
        v_poly_list.append(1.)
        v_polynomial = np.array(v_poly_list)

        # perform vignette correction
        # get coordinate grid across image, seem swapped because of transposed vignette
        x_dim, y_dim = self.raw().shape[1], self.raw().shape[0]
        x, y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))

        # meshgrid returns transposed arrays
        x = x.T
        y = y.T

        # compute matrix of distances from image center
        r = np.hypot((x - vignette_center_x), (y - vignette_center_y))

        # compute the vignette polynomial for each distance - we divide by the polynomial so that the
        # corrected image is image_corrected = image_original * vignetteCorrection
        vignette = 1. / np.polyval(v_polynomial, r)
        return vignette, x, y

    def undistorted_radiance(self, force_recompute=False):
        return self.undistorted(self.radiance(force_recompute))

    def undistorted_reflectance(self, irradiance=None, force_recompute=False):
        return self.undistorted(self.reflectance(irradiance, force_recompute))

    def plottable_vignette(self):
        return self.vignette()[0].T

    def cv2_distortion_coeff(self):
        # dist_coeffs = np.array(k[0],k[1],p[0],p[1],k[2]])
        return np.array(self.distortion_parameters)[[0, 1, 3, 4, 2]]

    # values in pp are in [mm], rescale to pixels
    def principal_point_px(self):
        center_x = self.principal_point[0] * self.focal_plane_resolution_px_per_mm[0]
        center_y = self.principal_point[1] * self.focal_plane_resolution_px_per_mm[1]
        return center_x, center_y

    def cv2_camera_matrix(self):
        center_x, center_y = self.principal_point_px()

        # set up camera matrix for cv2
        cam_mat = np.zeros((3, 3))
        cam_mat[0, 0] = self.focal_length * self.focal_plane_resolution_px_per_mm[0]
        cam_mat[1, 1] = self.focal_length * self.focal_plane_resolution_px_per_mm[1]
        cam_mat[2, 2] = 1.0
        cam_mat[0, 2] = center_x
        cam_mat[1, 2] = center_y

        # set up distortion coefficients for cv2
        return cam_mat

    def rig_xy_offset_in_px(self):
        pixel_pitch_mm_x = 1.0 / self.focal_plane_resolution_px_per_mm[0]
        pixel_pitch_mm_y = 1.0 / self.focal_plane_resolution_px_per_mm[1]
        px_fov_x = 2.0 * math.atan2(pixel_pitch_mm_x / 2.0, self.focal_length)
        px_fov_y = 2.0 * math.atan2(pixel_pitch_mm_y / 2.0, self.focal_length)
        t_x = math.radians(self.rig_relatives[0]) / px_fov_x
        t_y = math.radians(self.rig_relatives[1]) / px_fov_y
        return t_x, t_y

    def undistorted(self, image):
        """Return the undistorted image from input Image."""
        # If we have already undistorted the same source, just return that here.
        # Otherwise, lazy compute the undistorted image.
        if self.__undistorted_source is not None and image.data == self.__undistorted_source.data:
            return self.__undistorted_image

        self.__undistorted_source = image

        new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(self.cv2_camera_matrix(),
                                                       self.cv2_distortion_coeff(),
                                                       self.size(),
                                                       1)
        map1, map2 = cv2.initUndistortRectifyMap(self.cv2_camera_matrix(),
                                                 self.cv2_distortion_coeff(),
                                                 np.eye(3),
                                                 new_cam_mat,
                                                 self.size(),
                                                 cv2.CV_32F)  # cv2.CV_32F for 32 bit floats
        # compute the undistorted 16 bit image
        self.__undistorted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        return self.__undistorted_image

    def plot_raw(self, title=None, fig_size=None, *kwargs):
        """
        Create a single plot of the raw image.
        :param title: str to be used in plot title
        :param fig_size: 2 Tuple of ints for figure dimensions
        :param kwargs: e.g. boolean show, s system file_path
        :return: fig, axis of plot
        """
        if title is None:
            title = '{} Band {} Raw DN'.format(self.band_name, self.band_index)
        return plotutils.plot_with_color_bar(self.raw(), title=title, fig_size=fig_size, *kwargs)

    def plot_intensity(self, title=None, fig_size=None, *kwargs):
        """
        Create a single plot of the image converted to uncalibrated intensity.
        :param title: str to be used in plot title
        :param fig_size: 2 Tuple of ints for figure dimensions
        :param kwargs: e.g. boolean show, s system file_path
        :return: fig, axis of plot
        """
        if title is None:
            title = '{} Band {} Intensity (DN*sec)'.format(self.band_name, self.band_index)
        return plotutils.plot_with_color_bar(self.intensity(), title=title, fig_size=fig_size, *kwargs)

    def plot_radiance(self, title=None, fig_size=None, *kwargs):
        """
        Create a single plot of the image converted to radiance.
        :param title: str to be used in plot title
        :param fig_size: 2 Tuple of ints for figure dimensions
        :param kwargs: e.g. boolean show, s system file_path
        :return: fig, axis of plot
        """
        if title is None:
            title = '{} Band {} Radiance'.format(self.band_name, self.band_index)
        return plotutils.plot_with_color_bar(self.radiance(), title=title, fig_size=fig_size, *kwargs)

    def plot_vignette(self, title=None, fig_size=None, *kwargs):
        """
        Create a single plot of the vignette.
        :param title: str to be used in plot title
        :param fig_size: 2 Tuple of ints for figure dimensions
        :param kwargs: e.g. boolean show, s system file_path
        :return: fig, axis of plot
        """
        if title is None:
            title = '{} Band {} Vignette'.format(self.band_name, self.band_index)
        return plotutils.plot_with_color_bar(self.plottable_vignette(), title=title, fig_size=fig_size, *kwargs)

    def plot_undistorted_radiance(self, title=None, fig_size=None, *kwargs):
        """
        Create a single plot of the undistorted radiance.
        :param title: str to be used in plot title
        :param fig_size: 2 Tuple of ints for figure dimensions
        :param kwargs: e.g. boolean show, s system file_path
        :return: fig, axis of plot
        """
        if title is None:
            title = '{} Band {} Undistorted Radiance'.format(self.band_name, self.band_index)
        return plotutils.plot_with_color_bar(self.undistorted(self.radiance()), title=title, fig_size=fig_size, *kwargs)

    def plot_all(self, fig_size=(13, 10), *kwargs):
        """
        Plot raw, plottable vignette, radiance, and undistorted radiance.
        :param fig_size: 2 Tuple of ints for figure dimensions
        :param kwargs: e.g. boolean show, str system file_path
        """
        plots = [self.raw(), self.plottable_vignette(), self.radiance(), self.undistorted(self.radiance())]
        plot_types = ['Raw', 'Vignette', 'Radiance', 'Undistorted Radiance']
        titles = ['{} Band {} {}'.format(str(self.band_name), str(self.band_index), tpe)
                  for tpe in plot_types]
        plotutils.subplot_with_color_bar(2, 2, plots, titles, fig_size=fig_size, *kwargs)

    def get_homography(self, ref, r=None, t=None):
        """
        Get the homography that maps from this image to the reference image. If we have externally supplied
        rotations/translations for the rig use these. Otherwise use the rig-relatives intrinsic to the image.
        :param ref:
        :param r:
        :param t:
        :return:
        """

        if r is None:
            r = rotations_degrees_to_rotation_matrix(self.rig_relatives)
        if t is None:
            t = np.zeros(3)

        a = np.zeros((4, 4))
        a[0:3, 0:3] = r
        a[0:3, 3] = t
        a[3, 3] = 1.
        c, _ = cv2.getOptimalNewCameraMatrix(self.cv2_camera_matrix(),
                                             self.cv2_distortion_coeff(),
                                             self.size(), 1)
        cr, _ = cv2.getOptimalNewCameraMatrix(ref.cv2_camera_matrix(),
                                              ref.cv2_distortion_coeff(),
                                              ref.size(), 1)
        cc = np.zeros((4, 4))
        cc[0:3, 0:3] = c
        cc[3, 3] = 1.
        ccr = np.zeros((4, 4))
        ccr[0:3, 0:3] = cr
        ccr[3, 3] = 1.

        b = np.array(np.dot(ccr, np.dot(a, np.linalg.inv(cc))))
        b[:, 2] = b[:, 2] - b[:, 3]
        b = b[0:3, 0:3]
        b = b / b[2, 2]
        return np.array(b)
