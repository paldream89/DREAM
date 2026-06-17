# -*- coding: utf-8 -*-
"""
Single-frame point drawing from STORM bin file.

Features:
- choose one .bin file with wx GUI
- draw points from a selected frame using x_start / y_start
- save PNG file
- also pop up a window to display the generated image

Output image contains only:
- white background
- solid circular red points
- no axes, no border, no text

Coordinate range is fixed at 0~200 pixels by default.
Image resolution is controlled by output_scale.
"""

import os
import numpy as np
from ReadSTORMBin_XcYcZZcFrame import read_storm_bin_XcYcZZcFrame
import wx

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def plot_single_frame_points(STORM_npdata, Single_frame,
                             plot_range_x=200, plot_range_y=200,
                             output_scale=4,
                             marker_radius_pixel=1.5,
                             marker_color=(255, 0, 0),
                             background_color=(255, 255, 255),
                             invert_y=True,
                             save_path='single_frame_plot.png'):
    """
    Plot x_start/y_start points from one selected frame.

    Parameters
    ----------
    STORM_npdata : structured numpy array
        Must contain fields: frame, x_start, y_start.
    Single_frame : int
        Selected frame number.
    plot_range_x, plot_range_y : int or float
        Coordinate range in original pixel unit. Default: 200 x 200.
    output_scale : int or float
        Output image scaling factor.
        Example: plot_range_x=200, output_scale=4 -> output 800 x 800 pixels.
    marker_radius_pixel : float
        Circle radius in original pixel unit.
    marker_color : tuple
        RGB color of points, default red.
    background_color : tuple
        RGB background color, default white.
    invert_y : bool
        True: image-style coordinate, y increases downward.
    save_path : str
        Output PNG path.
    """
    mask = (STORM_npdata['frame'] == int(Single_frame))
    x = STORM_npdata['x_start'][mask].astype(np.float32)
    y = STORM_npdata['y_start'][mask].astype(np.float32)

    inside = (x >= 0) & (x <= plot_range_x) & (y >= 0) & (y <= plot_range_y)
    x = x[inside]
    y = y[inside]

    W = int(round(plot_range_x * output_scale))
    H = int(round(plot_range_y * output_scale))
    radius = float(marker_radius_pixel) * float(output_scale)

    if PIL_AVAILABLE:
        img = Image.new('RGB', (W, H), background_color)
        draw = ImageDraw.Draw(img)

        for xx, yy in zip(x, y):
            px = float(xx) * float(output_scale)
            if invert_y:
                py = float(yy) * float(output_scale)
            else:
                py = (float(plot_range_y) - float(yy)) * float(output_scale)

            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=marker_color,
                outline=None
            )

        img.save(save_path)
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        dpi = 100
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1], facecolor='white')
        marker_size = (marker_radius_pixel * output_scale * 2.0) ** 2
        ax.scatter(x, y, s=marker_size, c=[np.array(marker_color) / 255.0],
                   marker='o', edgecolors='none', linewidths=0)
        ax.set_xlim(0, plot_range_x)
        ax.set_ylim(plot_range_y, 0) if invert_y else ax.set_ylim(0, plot_range_y)
        ax.set_aspect('equal', adjustable='box')
        ax.set_axis_off()
        fig.savefig(save_path, dpi=dpi, facecolor='white', edgecolor='white',
                    transparent=False, bbox_inches=None, pad_inches=0)
        plt.close(fig)

    return int(x.size), W, H


class ImageDisplayFrame(wx.Frame):
    def __init__(self, image_path, title='Single Frame Image'):
        bmp = wx.Bitmap(image_path, wx.BITMAP_TYPE_ANY)
        img_w = bmp.GetWidth()
        img_h = bmp.GetHeight()

        # Allow some margin but keep the frame near image size.
        super().__init__(None, title=title, size=(min(img_w + 20, 1400), min(img_h + 40, 1400)))

        panel = wx.ScrolledWindow(self)
        panel.SetScrollRate(10, 10)
        sizer = wx.BoxSizer(wx.VERTICAL)
        bitmap_ctrl = wx.StaticBitmap(panel, bitmap=bmp)
        sizer.Add(bitmap_ctrl, 0, wx.ALL, 0)
        panel.SetSizer(sizer)
        panel.SetVirtualSize((img_w, img_h))
        self.Centre()


def choose_bin_file():
    app = wx.App(False)
    frame = wx.Frame(None, -1, 'win.py')
    frame.SetSize(0, 0, 200, 50)

    openFileDialog = wx.FileDialog(
        frame,
        "Select one Bin file",
        "",
        "",
        "Bin files (*.bin)|*.bin",
        wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
    )

    if openFileDialog.ShowModal() != wx.ID_OK:
        openFileDialog.Destroy()
        frame.Destroy()
        app.Destroy()
        raise SystemExit("No bin file selected.")

    selected_bin_path = openFileDialog.GetPath()
    openFileDialog.Destroy()
    frame.Destroy()
    app.Destroy()
    return selected_bin_path


if __name__ == "__main__":
    selected_bin_path = choose_bin_file()

    # -------- Single frame plotting parameters --------
    Single_frame = 2638

    # Coordinate range in original pixel unit.
    single_frame_plot_range_x = 200
    single_frame_plot_range_y = 200

    # Output resolution factor.
    # output_scale=1 -> 200 x 200 px
    # output_scale=4 -> 800 x 800 px
    # output_scale=8 -> 1600 x 1600 px
    single_frame_output_scale = 4

    # Circle radius in original pixel unit.
    single_frame_marker_radius_pixel = 1.2

    # RGB color. Red = (255, 0, 0)
    single_frame_marker_color = (255, 0, 0)
    single_frame_background_color = (255, 255, 255)
    single_frame_invert_y = True

    frame_num, number_of_points, STORM_npdata = read_storm_bin_XcYcZZcFrame(selected_bin_path)

    save_path = selected_bin_path.replace('.bin', f'-SingleFrame{Single_frame}_{single_frame_output_scale}x.png')

    print('Single frame plotting...')
    n_points, W, H = plot_single_frame_points(
        STORM_npdata,
        Single_frame=Single_frame,
        plot_range_x=single_frame_plot_range_x,
        plot_range_y=single_frame_plot_range_y,
        output_scale=single_frame_output_scale,
        marker_radius_pixel=single_frame_marker_radius_pixel,
        marker_color=single_frame_marker_color,
        background_color=single_frame_background_color,
        invert_y=single_frame_invert_y,
        save_path=save_path,
    )

    print(f'Single frame image saved: {save_path}')
    print(f'Image size: {W} x {H} pixels')
    print(f'Coordinate range: 0-{single_frame_plot_range_x}, 0-{single_frame_plot_range_y} pixels')
    print(f'Number of points in frame {Single_frame}: {n_points}')

    # Show popup window with the saved image.
    if os.path.exists(save_path):
        app2 = wx.App(False)
        viewer = ImageDisplayFrame(save_path, title=f'Single Frame {Single_frame}')
        viewer.Show()
        app2.MainLoop()
