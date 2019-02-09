import numpy as np


def plotHist(tableDict, frame_range = None, value_range=None, xy_range=None, pixel_size=20, sigma=None, target_size=None):
    x = tableDict['x'][:]
    y = tableDict['y'][:]
    
    if frame_range:
        frame_idx = (tableDict['frame'] >= frame_range[0]) & (tableDict['frame'] < frame_range[1])
        x = x[frame_idx]
        y = y[frame_idx]
    else:
        x = x[1:-1]
        y = y[1:-1]
        
    if xy_range:
        xmin, xmax = xy_range[0]
        ymin, ymax = xy_range[1]
    else:
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    xedges = np.arange(xmin, xmax, pixel_size)
    yedges = np.arange(ymin, ymax, pixel_size)
    H, xedgesO, yedgesO = np.histogram2d(y, x, bins=(yedges, xedges))
    if target_size is not None:
        if H.shape[0] < target_size[0] or H.shape[1] < target_size[1]:
            H = np.pad(H, ((0, target_size[0] - H.shape[0]), (0, target_size[
                       1] - H.shape[1])), mode='constant', constant_values=0)

    if value_range:
        H = H.clip(value_range[0], value_range[1])
    if sigma:
        import scipy
        H = scipy.ndimage.filters.gaussian_filter(H, sigma=(sigma, sigma))

    return H
