#!/usr/bin/env python
# coding: utf-8
"""
MicaSense Plotting Utilities

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# noinspection PyDefaultArgument
def plot_with_color_bar(img, title=None, fig_size=None, v_min=None, v_max=None, plot_text=None, text_loc_x=0.05,
                        text_loc_y=0.05, font_size=8, prop={'alpha': 0.6}, show=True, file_path=''):
    """
    Plot an image with a color bar.
    :param font_size:
    :param text_loc_x:
    :param text_loc_y:
    :param prop:
    :param img:
    :param title:
    :param fig_size:
    :param v_min:
    :param v_max:
    :param show: boolean True to call plt.show()
    :param file_path: str file path to save plots
    :param plot_text:
    :return:
    """
    fig, axes = plt.subplots(1, 1, figsize=fig_size)
    rad2 = axes.imshow(img, vmin=v_min, vmax=v_max)
    axes.set_title(title)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(rad2, cax=cax)

    if plot_text is not None:
        # create annotation box
        axes.text(text_loc_x, text_loc_y, plot_text, transform=axes.transAxes, fontsize=font_size, bbox=prop)

    plt.tight_layout()
    if file_path != '':
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes


# noinspection PyDefaultArgument
def subplot_with_color_bar(rows, cols, images, titles=None, fig_size=None, plot_text=None, text_loc_x=0.05,
                           text_loc_y=0.05, font_size=8, prop={'size': 8, 'alpha': 0.6}, show=True, file_path=''):
    """
    Plot a set of images in subplots.
    :param font_size:
    :param text_loc_y:
    :param text_loc_x:
    :param plot_text:
    :param prop:
    :param rows:
    :param cols:
    :param images:
    :param titles:
    :param fig_size:
    :param show: boolean True to call plt.show()
    :param file_path: str file path to save plots
    :return:
    """
    fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
    for i in range(cols * rows):
        column = int(i % cols)
        row = int(i / cols)
        if i < len(images):
            rad = axes[row][column].imshow(images[i])
            if titles is not None:
                axes[row][column].set_title(titles[i])
            divider = make_axes_locatable(axes[row][column])
            cax = divider.append_axes("right", size="3%", pad=0.05)
            fig.colorbar(rad, cax=cax)
        else:
            axes[row, column].axis('off')

    if plot_text is not None:
        # create annotation box
        axes.text(text_loc_x, text_loc_y, plot_text, transform=axes.transAxes, fontsize=font_size, bbox=prop)

    plt.tight_layout()
    if file_path != '':
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes


def plot_overlay_with_color_bar(img_base, img_color, title=None, fig_size=None, v_min=None, v_max=None,
                                overlay_alpha=1.0, overlay_colormap='viridis', overlay_steps=None,
                                display_contours=False, contour_fmt=None, contour_steps=None, contour_alpha=None,
                                show=True, file_path=''):
    """
    Plot an image with a color bar.
    :param img_base:
    :param img_color:
    :param title:
    :param fig_size:
    :param v_min:
    :param v_max:
    :param overlay_alpha:
    :param overlay_colormap:
    :param overlay_steps:
    :param display_contours:
    :param contour_fmt:
    :param contour_steps:
    :param contour_alpha:
    :param show: boolean True to call plt.show()
    :param file_path: str file path to save plots
    :return:
    """
    fig, axis = plt.subplots(1, 1, figsize=fig_size, squeeze=False)
    base = axis[0][0].imshow(img_base)  # FIXME: variable not used?
    if overlay_steps is not None:
        overlay_colormap = cm.get_cmap(overlay_colormap, overlay_steps)
    rad2 = axis[0][0].imshow(img_color, vmin=v_min, vmax=v_max, alpha=overlay_alpha, cmap=overlay_colormap)
    if display_contours:
        if contour_steps is None:
            contour_steps = overlay_steps
        if contour_alpha is None:
            contour_alpha = overlay_alpha
        contour_cmap = cm.get_cmap(overlay_colormap, contour_steps)
        contour_list = np.arange(v_min, v_max, (v_max - v_min) / contour_steps)
        rad3 = axis[0][0].contour(img_color, contour_list, cmap=contour_cmap, alpha=contour_alpha)
        font_size = 8 + (max(fig_size) / 10) * 2
        axis[0][0].clabel(rad3, rad3.levels, inline=True, fontsize=font_size, fmt=contour_fmt)
    axis[0][0].set_title(title)
    divider = make_axes_locatable(axis[0][0])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(rad2, cax=cax)
    plt.tight_layout()
    if file_path != '':
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig, axis[0][0]


def subplot(rows, cols, images, titles=None, fig_size=None, show=True, file_path=''):
    """
    Plot a set of images in subplots.
    :param rows: int number of rows for matplotlib subplot
    :param cols: int number of columns for matplotlib subplot
    :param images: list [ndarray] image or [ndarray, ...] images
    :param titles: str plot title or [str, ...] plot titles. Title position corresponds to image position.
    :param fig_size: 2 int tuple for plot size
    :param show: boolean True to call plt.show()
    :param file_path: str file path to save plots
    :return:
    """
    fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
    for i in range(cols * rows):
        column = int(i % cols)
        row = int(i / cols)
        if i < len(images):
            axes[row][column].imshow(images[i])
            if titles is not None:
                axes[row][column].set_title(titles[i])
        else:
            axes[row, column].axis('off')
    plt.tight_layout()
    if file_path != '':
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes


def colormap(cmap):
    """
    Set the default plotting colormap. Could be one of 'gray, viridis, plasma, inferno, magma, nipy_spectral.
    :param cmap: str with colormap value
    """
    plt.set_cmap(cmap)


def plot_ned_vector3d(x, y, z, u=0, v=0, w=0, title=None, fig_size=(8, 5), show=True, file_path=''):
    """
    Create a 3d plot of a North-East-Down vector. XYZ is the (tip of the) vector, uvw is the base location of the
    vector.
    :param x:
    :param y:
    :param z:
    :param u:
    :param v:
    :param w:
    :param title:
    :param fig_size:
    :param file_path:
    :param show:
    :return:
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca(projection='3d')
    ax.quiver(u, v, w, x, y, z, color='r')
    ax.quiver(u, v, w, x, y, 0, color='b')
    ax.quiver(x, y, 0, 0, 0, z, color='g')

    ax.legend()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel("West - East")
    ax.set_ylabel("South - North")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if file_path != '':
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax
