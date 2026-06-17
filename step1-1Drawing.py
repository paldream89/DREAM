# -*- coding: utf-8 -*-
"""
Single-frame drawing from STORM bin file.

This script does three things for a selected frame:
1. Draw x_start / y_start points only.
2. Draw x_end / y_end points only.
3. Draw x_start / y_start and corresponding x_end / y_end, connected by solid lines.

For each plot:
- save PNG file
- pop up a wx window to display the image

Image style:
- white background
- no axes, no border, no text
- coordinate range fixed at 0~200 pixels by default
- output resolution controlled by output_scale
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


def _save_scatter_image(x, y,
                        plot_range_x=200, plot_range_y=200,
                        output_scale=4,
                        marker_radius_pixel=1.5,
                        marker_color=(255, 0, 0),
                        background_color=(255, 255, 255),
                        invert_y=True,
                        save_path='scatter_plot.png'):
    """Save one scatter-only image."""
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
            py = float(yy) * float(output_scale) if invert_y else (float(plot_range_y) - float(yy)) * float(output_scale)
            draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                         fill=marker_color, outline=None)

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


def _save_pair_image(x0, y0, x1, y1,
                     plot_range_x=200, plot_range_y=200,
                     output_scale=4,
                     start_marker_radius_pixel=1.5,
                     end_marker_radius_pixel=1.5,
                     start_marker_color=(255, 0, 0),
                     end_marker_color=(0, 0, 255),
                     line_width_pixel=0.3,
                     line_color=(0, 0, 0),
                     background_color=(255, 255, 255),
                     invert_y=True,
                     save_path='pair_plot.png'):
    """Save one image containing start points, end points, and connecting lines."""
    inside = ((x0 >= 0) & (x0 <= plot_range_x) & (y0 >= 0) & (y0 <= plot_range_y) &
              (x1 >= 0) & (x1 <= plot_range_x) & (y1 >= 0) & (y1 <= plot_range_y))
    x0 = x0[inside]
    y0 = y0[inside]
    x1 = x1[inside]
    y1 = y1[inside]

    W = int(round(plot_range_x * output_scale))
    H = int(round(plot_range_y * output_scale))
    r0 = float(start_marker_radius_pixel) * float(output_scale)
    r1 = float(end_marker_radius_pixel) * float(output_scale)
    line_w = max(1, int(round(float(line_width_pixel) * float(output_scale))))

    if PIL_AVAILABLE:
        img = Image.new('RGB', (W, H), background_color)
        draw = ImageDraw.Draw(img)

        for xa, ya, xb, yb in zip(x0, y0, x1, y1):
            px0 = float(xa) * float(output_scale)
            py0 = float(ya) * float(output_scale) if invert_y else (float(plot_range_y) - float(ya)) * float(output_scale)
            px1 = float(xb) * float(output_scale)
            py1 = float(yb) * float(output_scale) if invert_y else (float(plot_range_y) - float(yb)) * float(output_scale)
            draw.line((px0, py0, px1, py1), fill=line_color, width=line_w)

        for xa, ya in zip(x0, y0):
            px0 = float(xa) * float(output_scale)
            py0 = float(ya) * float(output_scale) if invert_y else (float(plot_range_y) - float(ya)) * float(output_scale)
            draw.ellipse((px0 - r0, py0 - r0, px0 + r0, py0 + r0), fill=start_marker_color, outline=None)

        for xb, yb in zip(x1, y1):
            px1 = float(xb) * float(output_scale)
            py1 = float(yb) * float(output_scale) if invert_y else (float(plot_range_y) - float(yb)) * float(output_scale)
            draw.ellipse((px1 - r1, py1 - r1, px1 + r1, py1 + r1), fill=end_marker_color, outline=None)

        img.save(save_path)
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        dpi = 100
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1], facecolor='white')
        for xa, ya, xb, yb in zip(x0, y0, x1, y1):
            ax.plot([xa, xb], [ya, yb], color=np.array(line_color) / 255.0,
                    linewidth=max(0.5, line_width_pixel * output_scale / 2.0), solid_capstyle='round')
        size0 = (start_marker_radius_pixel * output_scale * 2.0) ** 2
        size1 = (end_marker_radius_pixel * output_scale * 2.0) ** 2
        ax.scatter(x0, y0, s=size0, c=[np.array(start_marker_color) / 255.0], marker='o', edgecolors='none', linewidths=0)
        ax.scatter(x1, y1, s=size1, c=[np.array(end_marker_color) / 255.0], marker='o', edgecolors='none', linewidths=0)
        ax.set_xlim(0, plot_range_x)
        ax.set_ylim(plot_range_y, 0) if invert_y else ax.set_ylim(0, plot_range_y)
        ax.set_aspect('equal', adjustable='box')
        ax.set_axis_off()
        fig.savefig(save_path, dpi=dpi, facecolor='white', edgecolor='white',
                    transparent=False, bbox_inches=None, pad_inches=0)
        plt.close(fig)

    return int(x0.size), W, H


def plot_single_frame_start(STORM_npdata, Single_frame, **kwargs):
    mask = (STORM_npdata['frame'] == int(Single_frame))
    x = STORM_npdata['x_start'][mask].astype(np.float32)
    y = STORM_npdata['y_start'][mask].astype(np.float32)
    return _save_scatter_image(x, y, **kwargs)


def plot_single_frame_end(STORM_npdata, Single_frame, **kwargs):
    mask = (STORM_npdata['frame'] == int(Single_frame))
    x = STORM_npdata['x_end'][mask].astype(np.float32)
    y = STORM_npdata['y_end'][mask].astype(np.float32)
    return _save_scatter_image(x, y, **kwargs)


def plot_single_frame_pairs(STORM_npdata, Single_frame, **kwargs):
    mask = (STORM_npdata['frame'] == int(Single_frame))
    x0 = STORM_npdata['x_start'][mask].astype(np.float32)
    y0 = STORM_npdata['y_start'][mask].astype(np.float32)
    x1 = STORM_npdata['x_end'][mask].astype(np.float32)
    y1 = STORM_npdata['y_end'][mask].astype(np.float32)
    return _save_pair_image(x0, y0, x1, y1, **kwargs)


class ImageDisplayFrame(wx.Frame):
    def __init__(self, image_path, title='Image'):
        bmp = wx.Bitmap(image_path, wx.BITMAP_TYPE_ANY)
        img_w = bmp.GetWidth()
        img_h = bmp.GetHeight()
        super().__init__(None, title=title,
                         size=(min(img_w + 20, 1400), min(img_h + 40, 1400)))
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

    # -------- plotting parameters --------
    Single_frame = 2637

    # coordinate range in original pixel unit
    single_frame_plot_range_x = 200
    single_frame_plot_range_y = 200

    # resolution factor: 1 -> 200x200 px, 4 -> 800x800 px
    single_frame_output_scale = 4

    # first plot: start points
    start_marker_radius_pixel = 1.2
    start_marker_color = (255, 0, 0)

    # second plot: end points
    end_marker_radius_pixel = 1.2
    end_marker_color = (0, 0, 255)

    # third plot: line + both endpoints
    line_width_pixel = 0.5
    line_color = (0, 0, 0)

    background_color = (255, 255, 255)
    invert_y = True

    frame_num, number_of_points, STORM_npdata = read_storm_bin_XcYcZZcFrame(selected_bin_path)

    save_path_start = selected_bin_path.replace('.bin', f'-SingleFrame{Single_frame}_start_{single_frame_output_scale}x.png')
    save_path_end = selected_bin_path.replace('.bin', f'-SingleFrame{Single_frame}_end_{single_frame_output_scale}x.png')
    save_path_pair = selected_bin_path.replace('.bin', f'-SingleFrame{Single_frame}_pair_{single_frame_output_scale}x.png')

    print('Single frame plotting: start points...')
    n_start, W1, H1 = plot_single_frame_start(
        STORM_npdata,
        Single_frame=Single_frame,
        plot_range_x=single_frame_plot_range_x,
        plot_range_y=single_frame_plot_range_y,
        output_scale=single_frame_output_scale,
        marker_radius_pixel=start_marker_radius_pixel,
        marker_color=start_marker_color,
        background_color=background_color,
        invert_y=invert_y,
        save_path=save_path_start,
    )

    print('Single frame plotting: end points...')
    n_end, W2, H2 = plot_single_frame_end(
        STORM_npdata,
        Single_frame=Single_frame,
        plot_range_x=single_frame_plot_range_x,
        plot_range_y=single_frame_plot_range_y,
        output_scale=single_frame_output_scale,
        marker_radius_pixel=end_marker_radius_pixel,
        marker_color=end_marker_color,
        background_color=background_color,
        invert_y=invert_y,
        save_path=save_path_end,
    )

    print('Single frame plotting: paired start/end with lines...')
    n_pair, W3, H3 = plot_single_frame_pairs(
        STORM_npdata,
        Single_frame=Single_frame,
        plot_range_x=single_frame_plot_range_x,
        plot_range_y=single_frame_plot_range_y,
        output_scale=single_frame_output_scale,
        start_marker_radius_pixel=start_marker_radius_pixel,
        end_marker_radius_pixel=end_marker_radius_pixel,
        start_marker_color=start_marker_color,
        end_marker_color=end_marker_color,
        line_width_pixel=line_width_pixel,
        line_color=line_color,
        background_color=background_color,
        invert_y=invert_y,
        save_path=save_path_pair,
    )

    print(f'Start image saved: {save_path_start}')
    print(f'End image saved:   {save_path_end}')
    print(f'Pair image saved:  {save_path_pair}')
    print(f'Image size: {W1} x {H1} pixels')
    print(f'Coordinate range: 0-{single_frame_plot_range_x}, 0-{single_frame_plot_range_y} pixels')
    print(f'Number of start points in frame {Single_frame}: {n_start}')
    print(f'Number of end points in frame {Single_frame}:   {n_end}')
    print(f'Number of connected pairs in frame {Single_frame}: {n_pair}')

    # Show popup windows with the saved images.
    paths_and_titles = [
        (save_path_start, f'Single Frame {Single_frame} - Start'),
        (save_path_end, f'Single Frame {Single_frame} - End'),
        (save_path_pair, f'Single Frame {Single_frame} - Pair'),
    ]
    existing = [(p, t) for p, t in paths_and_titles if os.path.exists(p)]
    if existing:
        app2 = wx.App(False)
        frames = []
        for p, t in existing:
            viewer = ImageDisplayFrame(p, title=t)
            viewer.Show()
            frames.append(viewer)
        app2.MainLoop()
