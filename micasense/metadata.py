#!/usr/bin/env python
# coding: utf-8
"""
RedEdge Metadata Management Utilities

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
from datetime import datetime, timedelta

import exiftool
import pytz


class Metadata(object):
    """Container for MicaSense Image metadata."""

    def __init__(self, file_path, exiftool_obj=None):
        if not os.path.isfile(file_path):
            raise IOError("Provided path is not a file: {}".format(file_path))

        # assume Exiftool is in PATH
        if isinstance(exiftool_obj, exiftool.ExifTool):
            self.exif = exiftool_obj.get_metadata(file_path)
        else:
            with exiftool.ExifTool() as exift:
                self.exif = exift.get_metadata(file_path)

    def get_all(self):
        """
        Get all extracted metadata values.
        :return: dict {str:<Any>} of all Image exif tags
        """
        return self.exif

    def get_item(self, item, index=None):
        """
        Get metadata item by Tag:Parameter.
        :param item: str metadata item name.
                    e.g. XMP:VignettingPolynomial
        :param index: int positional index for items with multiple values.
                    e.g. XMP:RigRelatives -0.020453, 0.066033, -0.063915
        :return: value at metadata Tag:Parameter. Else None if value not found.
        """
        val = None
        try:
            val = self.exif[item]
            if index is not None:
                if isinstance(val, str) and len(val.split(',')) > 1:
                    val = val.split(',')
                val = val[index]
        except KeyError:
            # print("Item {} not found.".format(item))
            pass
        except IndexError:
            print("Item {0} is length {1}, index {2} is outside this range.".format(
                item,
                len(self.exif[item]),
                index))
        return val

    def size(self, item):
        """
        Get the size (length) of a metadata item.
        :param item: str metadata item name.
                    e.g. XMP:VignettingPolynomial
        :return: int len() of the item. Return 0 if val is None.
        """
        val = self.get_item(item)
        if isinstance(val, str) and len(val.split(',')) > 1:
            val = val.split(',')
        if val is not None:
            return len(val)
        else:
            return 0

    def print_all(self):
        """Print all metadata Tag:Value."""
        for item in self.get_all():
            print("{}: {}".format(item, self.get_item(item)))

    def dls_present(self):
        """
        Determine if DLS was used during image capture.
        :return: bool True if DLS metadata exists, else False.
        """
        return self.get_item("XMP:Irradiance") is not None \
               or self.get_item("XMP:HorizontalIrradiance") is not None \
               or self.get_item("XMP:DirectIrradiance") is not None

    def supports_radiometric_calibration(self):
        """
        Determine if metadata supports radiometric calibration.
        :return: bool
        """
        if (self.get_item('XMP:RadiometricCalibration')) is None:
            return False
        return True

    def radiometric_cal(self):
        nelem = self.size('XMP:RadiometricCalibration')
        return [float(self.get_item('XMP:RadiometricCalibration', i)) for i in range(nelem)]

    def position(self):
        """
        Get the WGS-84 (latitude, longitude, altitude) tuple as signed decimal degrees.
        :return: tuple (lat, lon, alt)
        """

        lat = self.get_item('EXIF:GPSLatitude')
        latref = self.get_item('EXIF:GPSLatitudeRef')
        if latref == 'S':
            lat *= -1.0

        lon = self.get_item('EXIF:GPSLongitude')
        lonref = self.get_item('EXIF:GPSLongitudeRef')
        if lonref == 'W':
            lon *= -1.0

        alt = self.get_item('EXIF:GPSAltitude')
        return lat, lon, alt

    def utc_time(self):
        """Get the timezone-aware datetime of the image capture."""
        str_time = self.get_item('EXIF:DateTimeOriginal')
        utc_time = datetime.strptime(str_time, "%Y:%m:%d %H:%M:%S")
        subsec = int(self.get_item('EXIF:SubSecTime'))
        negative = 1.0
        if subsec < 0:
            negative = -1.0
            subsec *= -1.0
        subsec = float('0.{}'.format(int(subsec)))
        subsec *= negative
        ms = subsec * 1e3
        utc_time += timedelta(milliseconds=ms)
        timezone = pytz.timezone('UTC')
        utc_time = timezone.localize(utc_time)
        return utc_time

    def dls_pose(self):
        """Get DLS pose as local earth-fixed yaw, pitch, roll in radians."""
        if self.get_item('XMP:Yaw') is not None:
            return float(self.get_item('XMP:Yaw')), float(self.get_item('XMP:Pitch')), float(self.get_item('XMP:Roll'))
        else:
            return 0.0, 0.0, 0.0  # if metadata doesn't exist, return 0.0 for all

    def rig_relatives(self):
        """
        Get Rig Relatives data if present.
        :return: list of Rig Relatives values. None if not present
        """
        if self.get_item('XMP:RigRelatives') is not None:
            nelem = self.size('XMP:RigRelatives')
            return [float(self.get_item('XMP:RigRelatives', i)) for i in range(nelem)]
        else:
            return None

    def capture_id(self):
        """Get CaptureId."""
        return self.get_item('XMP:CaptureId')

    def flight_id(self):
        """Get FlightId."""
        return self.get_item('XMP:FlightId')

    def camera_make(self):
        """Get camera make."""
        return self.get_item('EXIF:Make')

    def camera_model(self):
        """Get camera model."""
        return self.get_item('EXIF:Model')

    def firmware_version(self):
        """Get firmware version."""
        return self.get_item('EXIF:Software')

    def band_name(self):
        """Get BandName."""
        return self.get_item('XMP:BandName')

    def band_index(self):
        """Get band index."""
        return self.get_item('XMP:RigCameraIndex')

    def exposure(self):
        """Get exposure time."""
        exp = self.get_item('EXIF:ExposureTime')
        # correct for incorrect exposure in some legacy RedEdge firmware versions
        if self.camera_model() != "Altum":
            if math.fabs(exp - (1.0 / 6329.0)) < 1e-6:
                exp = 0.000274
        return exp

    def gain(self):
        """Get gain as ISOSpeed / 100.0."""
        return self.get_item('EXIF:ISOSpeed') / 100.0

    def image_size(self):
        """Get ImageWidth, ImageHeight."""
        return self.get_item('EXIF:ImageWidth'), self.get_item('EXIF:ImageHeight')

    def center_wavelength(self):
        """Get center wavelength."""
        return self.get_item('XMP:CentralWavelength')

    def bandwidth(self):
        """Get bandwidth as Full Width at Half Maximum."""
        return self.get_item('XMP:WavelengthFWHM')

    def black_level(self):
        """Get average black level value as float. Else 0 if black levels don't exist in metadata."""
        if self.get_item('EXIF:BlackLevel') is None:
            return 0
        black_lvls = [float(pixel) for pixel in self.get_item('EXIF:BlackLevel').split(' ')]
        return math.fsum(black_lvls) / float(len(black_lvls))

    def dark_pixels(self):
        """Get the average of the optically covered pixel values.
        Note: these pixels are raw, and have not been radiometrically corrected.
        Use the black_level() method for all radiometric calibrations."""
        dark_pixels = [float(pixel) for pixel in self.get_item('XMP:DarkRowValue')]
        return math.fsum(dark_pixels) / float(len(dark_pixels))

    def bits_per_pixel(self):
        """Get the number of bits per pixel, which defines the maximum digital number value in the image."""
        return self.get_item('EXIF:BitsPerSample')

    def vignette_center(self):
        """Get the vignette center in X and Y image coordinates."""
        nelem = self.size('XMP:VignettingCenter')
        return [float(self.get_item('XMP:VignettingCenter', i)) for i in range(nelem)]

    def vignette_polynomial(self):
        """Get the radial vignette polynomial in the order it's defined in the metadata."""
        nelem = self.size('XMP:VignettingPolynomial')
        return [float(self.get_item('XMP:VignettingPolynomial', i)) for i in range(nelem)]

    def distortion_parameters(self):
        """Get list of distortion parameters."""
        nelem = self.size('XMP:PerspectiveDistortion')
        return [float(self.get_item('XMP:PerspectiveDistortion', i)) for i in range(nelem)]

    def principal_point(self):
        """Get list of principal point coordinates (cx, cy) in millimeters."""
        return [float(item) for item in self.get_item('XMP:PrincipalPoint').split(',')]

    def focal_plane_resolution_px_per_mm(self):
        """Get focal plane X resolution and focal plane Y resolution."""
        fp_x_resolution = float(self.get_item('EXIF:FocalPlaneXResolution'))
        fp_y_resolution = float(self.get_item('EXIF:FocalPlaneYResolution'))
        return fp_x_resolution, fp_y_resolution

    def focal_length_mm(self):
        """Get focal length in millimeters."""
        if self.get_item('XMP:PerspectiveFocalLengthUnits') == 'mm':
            focal_length_mm = float(self.get_item('XMP:PerspectiveFocalLength'))
        else:
            focal_length_px = float(self.get_item('XMP:PerspectiveFocalLength'))
            focal_length_mm = focal_length_px / self.focal_plane_resolution_px_per_mm()[0]
        return focal_length_mm

    def focal_length_35_mm_eq(self):
        """Get 35 mm equivalent focal length."""
        return float(self.get_item('Composite:FocalLength35efl'))

    def irradiance_scale_factor(self):
        """Get the calibration scale factor for the irradiance measurements from the image metadata.
        Due to calibration differences between DLS1 and DLS2, we need to account for a scale factor change in their
        respective units. This scale factor is pulled from the image metadata, or, if the metadata doesn't give us the
        scale, we assume one based on a known combination of tags."""
        if self.get_item('XMP:IrradianceScaleToSIUnits') is not None:
            # the metadata contains the scale
            scale_factor = self.__float_or_zero(self.get_item('XMP:IrradianceScaleToSIUnits'))
        elif self.get_item('XMP:HorizontalIrradiance') is not None:
            # DLS2 but the metadata is missing the scale, assume 0.01. uW/cm^2/nm
            scale_factor = 0.01
        else:
            # DLS1, so we use a scale of 1
            scale_factor = 1.0
        return scale_factor

    def horizontal_irradiance_valid(self):
        """Defines if horizontal irradiance tag contains a value that can be trusted. Some firmware versions had a bug
        whereby the direct and scattered irradiance were correct, but the horizontal irradiance was calculated
        incorrectly."""
        if self.get_item('XMP:HorizontalIrradiance') is None:
            return False
        from packaging import version
        version_string = self.firmware_version().strip('v')
        if self.camera_model() == "Altum":
            good_version = "1.2.3"
        elif self.camera_model() == 'RedEdge' or self.camera_model() == 'RedEdge-M':
            good_version = "5.1.7"
        else:
            raise ValueError("Camera model is required to be RedEdge or Altum, not {} ".format(self.camera_model()))
        return version.parse(version_string) >= version.parse(good_version)

    def spectral_irradiance(self):
        """Raw spectral irradiance measured by an irradiance sensor. Calibrated to W/m^2/nm using
        irradiance_scale_factor, but not corrected for angles."""
        return self.__float_or_zero(self.get_item('XMP:SpectralIrradiance')) * self.irradiance_scale_factor()

    def horizontal_irradiance(self):
        """Horizontal irradiance at the earth's surface below the DLS on the plane normal to the gravity vector at the
        location (local flat plane spectral irradiance)."""
        return self.__float_or_zero(self.get_item('XMP:HorizontalIrradiance')) * self.irradiance_scale_factor()

    def scattered_irradiance(self):
        """Scattered component of the spectral irradiance."""
        return self.__float_or_zero(self.get_item('XMP:ScatteredIrradiance')) * self.irradiance_scale_factor()

    def direct_irradiance(self):
        """Direct component of the spectral irradiance on a plane normal to the vector towards the sun."""
        return self.__float_or_zero(self.get_item('XMP:DirectIrradiance')) * self.irradiance_scale_factor()

    def solar_azimuth(self):
        """Solar azimuth at the time of capture, as calculated by the camera system."""
        return self.__float_or_zero(self.get_item('XMP:SolarAzimuth'))

    def solar_elevation(self):
        """Solar elevation at the time of capture, as calculated by the camera system."""
        return self.__float_or_zero(self.get_item('XMP:SolarElevation'))

    def estimated_direct_vector(self):
        """Estimated direct light vector relative to the DLS2 reference frame."""
        if self.get_item('XMP:EstimatedDirectLightVector') is not None:
            return [self.__float_or_zero(item) for item in self.get_item('XMP:EstimatedDirectLightVector')]
        else:
            return None

    def auto_calibration_image(self):
        """True if this image is an auto-calibration image, where the camera has found and identified a calibration
        panel."""
        cal_tag = self.get_item('XMP:CalibrationPicture')
        return cal_tag is not None and \
               cal_tag == 2 and \
               self.panel_albedo() is not None and \
               self.panel_region() is not None and \
               self.panel_serial() is not None

    def panel_albedo(self):
        """Surface albedo of the active portion of the reflectance panel as calculated by the camera (usually from the
        information in the panel QR code)."""
        albedo = self.get_item('XMP:Albedo')
        if albedo is not None:
            return self.__float_or_zero(albedo)
        return albedo

    def panel_region(self):
        """A 4-tuple containing image x,y coordinates of the panel active area."""
        if self.get_item('XMP:ReflectArea') is not None:
            coords = [int(item) for item in self.get_item('XMP:ReflectArea').split(',')]
            return list(zip(coords[0::2], coords[1::2]))
        else:
            return None

    def panel_serial(self):
        """The panel serial number as extracted from the image by the camera."""
        return self.get_item('XMP:PanelSerial')

    def get_gps_accuracy(self):
        """Get GPS XY and GPS Z accuracy in meters."""
        return self.get_item('XMP:GPSXYAccuracy'), self.get_item('XMP:GPSZAccuracy')

    @staticmethod
    def __float_or_zero(s):
        if s is not None:
            return float(s)
        else:
            return 0.0

