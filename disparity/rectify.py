import math
import numpy as np
from skimage import transform


def redefine_bounds(imgs, H):
    # calculate new bounds such that all pixels within the image pairs
    # have positive pixel coordinates
    x_min, y_min = np.Inf, np.Inf
    x_max, y_max = -np.Inf, -np.Inf
    x_min_idx, y_min_idx = 0, 0
    x_max_idx, y_max_idx = 0, 0
    for idx, img in enumerate(imgs):
        nrows1, ncols1, nbands1 = img.shape

        c = 1
        x_tl = [0, 0, c]
        x_tr = [0, nrows1, c]
        x_bl = [ncols1, 0, c]
        x_br = [ncols1, nrows1, c]
        x_corners = [x_tl, x_tr, x_bl, x_br]

        for x_corner in x_corners:
            x_corner = np.array(x_corner)
            corner = np.linalg.inv(H[idx])@x_corner
            x, y, _ = corner / corner[-1]
            if x < x_min:
                x_min = x
                x_min_idx = idx
            if y < y_min:
                y_min = y
                y_min_idx = idx
            if x > x_max:
                x_max = x
                x_max_idx = idx
            if y > y_max:
                y_max = y
                y_max_idx = idx
    bounds = (x_min, x_max, y_min, y_max)
    bounds_idx = (x_min_idx, x_max_idx, y_min_idx, y_max_idx)
    return bounds, bounds_idx


def rectify_stereopair(imgarray1, imgarray2, imgparams1, imgparams2, DF):
    """
    Rectify two downsampled image pairs.

    :param img1: numpy array of the left image
    :param img2: numpy array of the right image
    :param imgparams1: camera parameters of the left image
    :param imgparams2: camera parameters of the right image
    :param DF: downsampling factor constant
    :return: transformed image arrays and rectification parameters
    """
    R1 = imgparams1["rotation_matrix"]
    R2 = imgparams2["rotation_matrix"]
    Z1 = np.array([imgparams1["X"], imgparams1["Y"], imgparams1["Z"]])
    Z2 = np.array([imgparams2["X"], imgparams2["Y"], imgparams2["Z"]])
    K1 = imgparams1["camera_matrix"]
    K2 = imgparams2["camera_matrix"]
    # same camera matrix for all images
    K = K1
    K = K / DF
    K[-1, -1] = K[-1, -1] * DF
    R = np.zeros(R1.shape, dtype=np.float64)
    B_2_1 = Z2 - Z1
    R[0] = B_2_1 / np.linalg.norm(B_2_1)
    C1 = np.concatenate((np.eye(3), -np.expand_dims(Z1, axis=1)), axis=1)
    C2 = np.concatenate((np.eye(3), -np.expand_dims(Z2, axis=1)), axis=1)
    P1 = K@R1@C1
    P2 = K@R2@C2
    d1 = -P1[2,:3]
    d2 = -P2[2,:3]
    d_mean = (d1 / np.linalg.norm(d1)) + (d2 / np.linalg.norm(d2))
    d_mean = d_mean / np.linalg.norm(d_mean)
    d_y = np.cross(B_2_1, d_mean)
    d_z = np.cross(B_2_1, d_y)
    R[1] = d_y / np.linalg.norm(d_y)
    R[2] = d_z / np.linalg.norm(d_z)

    H1 = K@R1@R.T@np.linalg.inv(K)
    H2 = K@R2@R.T@np.linalg.inv(K)

    # homographies to warp image pair to a stereo normal pair
    H = [H1, H2]
    b, b_idx = redefine_bounds([imgarray1, imgarray2], H)
    outshape = (math.ceil(b[3]-b[2]), math.ceil(b[1]-b[0]))
    shifts = (math.ceil(b[0]), math.ceil(b[2]))
    H_shift = np.array([[1, 0, shifts[0]],
                        [0, 1, shifts[1]],
                        [0, 0, 1 ]])

    tform1 = transform.ProjectiveTransform(matrix=H1@H_shift)
    tform2 = transform.ProjectiveTransform(matrix=H2@H_shift)

    tf_img1 = transform.warp(imgarray1, tform1, output_shape=outshape)
    tf_img2 = transform.warp(imgarray2, tform2, output_shape=outshape)
    rect_params = {"H1": H1, "H2": H2, "H_shift": H_shift, "K": K, "R": R}
    return {"image_pairs": [tf_img1, tf_img2], "rect_params": rect_params}