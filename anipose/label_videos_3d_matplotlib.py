import numpy as np
import pandas as pd
import os
from glob import glob
from collections import defaultdict
import skvideo.io
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting
from matplotlib.pyplot import get_cmap

# These functions are assumed to be defined in your project.
from .common import make_process_fun, get_nframes, get_video_name, get_video_params, get_data_length, natural_keys

def connect(ax, points, bps, bp_dict, color):
    """
    Connect a set of points using a line on the provided Matplotlib axis.
    """
    ixs = [bp_dict[bp] for bp in bps]
    ax.plot(points[ixs, 0], points[ixs, 1], points[ixs, 2],
            color=color, linewidth=2)

def connect_all(ax, points, scheme, bp_dict, cmap):
    """
    Connect all body part segments as specified in the scheme.
    """
    for i, bps in enumerate(scheme):
        connect(ax, points, bps, bp_dict, color=cmap(i)[:3])

def visualize_labels(config, labels_fname, outname, fps=300):
    """
    Visualizes 3D skeleton labels using Matplotlib and writes the output to a video file.
    """
    try:
        scheme = config['labeling']['scheme']
    except KeyError:
        scheme = []
    
    data = pd.read_csv(labels_fname)

    print(f"{data=}")

    cols = [x for x in data.columns if '_error' in x]
    
    if len(scheme) == 0:
        bodyparts = [c.replace('_error', '') for c in cols]
    else:
        bodyparts = sorted(set([x for dx in scheme for x in dx]))
    
    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))
    
    # Create a numpy array of shape (n_bodyparts, n_frames, 3)
    all_points = np.array([
        data.loc[:, (bp + '_x', bp + '_y', bp + '_z')].values 
        for bp in bodyparts
    ], dtype='float64')
    
    # Compute fixed axis limits based on the entire dataset (using 5th and 95th percentiles)
    all_points_flat = all_points.reshape(-1, 3)
    valid = ~np.isnan(all_points_flat[:, 0])
    if np.sum(valid) < 10:
        print('Too few valid points to plot, skipping...')
        return
    lim_min = np.nanpercentile(all_points_flat[valid], 5, axis=0)
    lim_max = np.nanpercentile(all_points_flat[valid], 95, axis=0)
    
    # Setup the video writer
    writer = skvideo.io.FFmpegWriter(outname, inputdict={
        '-framerate': str(fps),
    }, outputdict={
        '-vcodec': 'h264', '-qp': '28', '-pix_fmt': 'yuv420p'
    })
    
    cmap = get_cmap('tab10')
    
    # Create a Matplotlib figure with a 3D axis
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set a fixed view (e.g., elevation and azimuth)
    fixed_elev = 20
    fixed_azim = -60
    
    n_frames = data.shape[0]
    for framenum in trange(n_frames, ncols=70):
        # Instead of clearing everything, we clear the axis and then reapply fixed limits and view
        ax.clear()
        
        # Reapply fixed axis limits so that the view remains fixed
        ax.set_xlim(lim_min[0], lim_max[0])
        ax.set_ylim(lim_min[1], lim_max[1])
        ax.set_zlim(lim_min[2], lim_max[2])
        
        # Ensure equal aspect ratio
        ax.set_box_aspect((1, 1, 1))

        # Reapply axis labels and view
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=fixed_elev, azim=fixed_azim)
        
        # Get the points for the current frame.
        # all_points shape is (n_bodyparts, n_frames, 3)
        if framenum < all_points.shape[1]:
            points = all_points[:, framenum]
        else:
            points = np.full((len(bodyparts), 3), np.nan)
        
        # Plot the skeleton points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   color='gray', s=20)
        # Draw the connecting lines based on the scheme
        connect_all(ax, points, scheme, bp_dict, cmap)
        
        # Draw the canvas and extract the image as an array
        fig.canvas.draw()

        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        # Calculate actual dimensions: total_pixels = frame.size / 3
        total_pixels = frame.size // 3
        # Assuming a square canvas, the side length is:
        side = int(np.sqrt(total_pixels))
        frame = frame.reshape((side, side, 3))

        # frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        # frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.writeFrame(frame)
    
    plt.close(fig)
    writer.close()

def process_session(config, session_path, filtered=False):
    """
    Process a session by reading videos and corresponding CSV files,
    then visualizing and saving labeled 3D videos.
    """
    pipeline_videos_raw = config['pipeline']['videos_raw']
    if filtered:
        pipeline_videos_labeled_3d = config['pipeline']['videos_labeled_3d_filter']
        pipeline_3d = config['pipeline']['pose_3d_filter']
    else:
        pipeline_videos_labeled_3d = config['pipeline']['videos_labeled_3d']
        pipeline_3d = config['pipeline']['pose_3d']
    
    video_ext = config['video_extension']
    
    vid_fnames = glob(os.path.join(session_path,
                                   pipeline_videos_raw, "*." + video_ext))
    orig_fnames = defaultdict(list)
    for vid in vid_fnames:
        vidname = get_video_name(config, vid)
        orig_fnames[vidname].append(vid)
    
    labels_fnames = glob(os.path.join(session_path, pipeline_3d, '*.csv'))
    labels_fnames = sorted(labels_fnames, key=natural_keys)
    
    outdir = os.path.join(session_path, pipeline_videos_labeled_3d)
    if len(labels_fnames) > 0:
        os.makedirs(outdir, exist_ok=True)
    
    for fname in labels_fnames:
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]
        out_fname = os.path.join(outdir, basename + '.mp4')
    
        if os.path.exists(out_fname) and abs(get_nframes(out_fname) - get_data_length(fname)) < 100:
            continue
        print("Processing:", out_fname)
        some_vid = orig_fnames[basename][0]
        params = get_video_params(some_vid)
        visualize_labels(config, fname, out_fname, params['fps'])

# Create processing functions using the provided common.make_process_fun
label_videos_3d_all = make_process_fun(process_session, filtered=False)
label_videos_3d_filtered_all = make_process_fun(process_session, filtered=True)

if __name__ == '__main__':
    # Optionally add code to run a session directly
    print("script started")
