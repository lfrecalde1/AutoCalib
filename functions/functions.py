#!/usr/bin/evn python
import numpy as np
import cv2
import os
import argparse
from scipy.io import savemat
import matplotlib.pyplot as plt
import casadi as ca


def GetImagesNames(base_path):
    # Define valid image extensions
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

    # Extract image file names from the folder
    image_files = sorted(
        [f for f in os.listdir(base_path) if f.lower().endswith(valid_extensions)]
    )

    return image_files


def GetCorners(base_path, image_files):
    # Read each image in grayscale
    for image_name in image_files:
        image_path = os.path.join(base_path, image_name)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            print(f"Failed to load image: {image_name}")
        else:
            print(f"Loaded {image_name} with dimensions: {gray_image.shape}")

    return None


def ComputeCorners(base_path, pattern_size, image_files):
    # Size of the square inside the cheesboard
    nx = pattern_size[0] - 1
    ny = pattern_size[1] - 1
    dx = 0.0215
    dy = 0.0215

    # Get number of images
    num_images = len(image_files)

    # Compute point over the plane of the cheesboard
    x_vals, y_vals = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy)
    Z = np.zeros_like(x_vals)
    points2D = np.vstack((y_vals.ravel(), x_vals.ravel(), Z.ravel()))

    # Emptu vectors to save the data
    data_uv = np.ones((3, points2D.shape[1], num_images))
    data_xy = np.ones((3, points2D.shape[1], num_images))

    for k, image_name in enumerate(image_files):
        full_file_name = os.path.join(base_path, image_name)
        print(f"Reading and processing image: {full_file_name}")

        I = cv2.imread(full_file_name)
        if I is None:
            print(f"Failed to load image: {full_file_name}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

        pattern_size = (nx, ny)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Refine corner locations (optional, but recommended)
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            drawn_image = I.copy()
            cv2.drawChessboardCorners(drawn_image, pattern_size, corners, ret)

            image_points = corners.reshape(-1, 2).T  # shape becomes (2, num_points)
            u_points = image_points[0, :]
            v_points = image_points[1, :]

            u_points = u_points.reshape((ny, nx))
            v_points = v_points.reshape((ny, nx))
            u_points = u_points[:, ::-1]
            v_points = v_points[:, ::-1]
            u_points = u_points.reshape((1, nx * ny))
            v_points = v_points.reshape((1, nx * ny))

            data_uv[0, :, k] = u_points
            data_uv[1, :, k] = v_points
            data_xy[0:2, :, k] = points2D[0:2, :]
        else:
            print(f"Checkerboard not detected in image: {full_file_name}")

    savemat(
        "calibration_values_python.mat",
        {"data_uv_python": data_uv, "data_xy_python": data_xy},
    )
    print("Calibration data saved to calibration_values.mat")
    return data_uv, data_xy


def Normalization(X):
    x_mean = np.mean(X[0, :])
    y_mean = np.mean(X[1, :])

    # Compute the sample variances for x and y (using ddof=1 to mimic MATLAB's var behavior)
    x_var = np.var(X[0, :], ddof=1)
    y_var = np.var(X[1, :], ddof=1)

    # Compute scaling factors
    s_x = np.sqrt(2.0 / x_var)
    s_y = np.sqrt(2.0 / y_var)

    # Construct the normalization matrix
    H_norm = np.array([[s_x, 0, -s_x * x_mean], [0, s_y, -s_y * y_mean], [0, 0, 1]])

    return H_norm


def EstimatePose(H, A):
    R = np.zeros((3, 3, H.shape[2]))
    t = np.zeros((3, H.shape[2]))
    A_inv = np.linalg.pinv(A)

    for k in range(H.shape[2]):
        h_0 = H[:, 0, k]
        h_1 = H[:, 1, k]
        h_2 = H[:, 2, k]
        lambda_aux = 1 / np.linalg.norm(A_inv @ h_0)
        r_0 = lambda_aux * (A_inv @ h_0)
        r_1 = lambda_aux * (A_inv @ h_1)
        r_2 = np.cross(r_0, r_1)

        # Form the rotation matrix by stacking the basis vectors as columns
        R_aux = np.column_stack((r_0, r_1, r_2))

        # Perform the SVD of R and recompose to ensure orthogonality
        U_r, _, Vh_r = np.linalg.svd(R_aux)
        R[:, :, k] = U_r @ Vh_r  # Note: Vh_r is V transposed

        # Compute the translation vector
        t[:, k] = lambda_aux * (A_inv @ h_2)

    return R, t


def HomographyAnalytical(X, U):
    # Normalize the data. Since our normalization function expects 2 x n arrays,
    # we pass only the first two rows (x and y coordinates).
    H_X = Normalization(X[0:2, :])
    H_U = Normalization(U[0:2, :])

    # Transform the points into normalized coordinates.
    X_norm = H_X @ X
    U_norm = H_U @ U

    A = []
    n_points = U_norm.shape[1]
    for k in range(n_points):
        x_0 = X_norm[0, k]
        y_0 = X_norm[1, k]
        u_0 = U_norm[0, k]
        v_0 = U_norm[1, k]

        # Construct the two rows for this correspondence.
        aux_a = [-x_0, -y_0, -1, 0, 0, 0, u_0 * x_0, u_0 * y_0, u_0]
        aux_b = [0, 0, 0, -x_0, y_0, -1, v_0 * x_0, v_0 * y_0, v_0]

        A.append(aux_a)
        A.append(aux_b)

    A = np.array(A)

    _, _, Vh = np.linalg.svd(A)
    h_nomr = Vh[-1, :]

    H_norm = h_nomr.reshape((3, 3))

    H = np.linalg.pinv(H_U) @ H_norm @ H_X
    return H


def EstimateHomographyCasadi(pts1, pts2, H_init):
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    N = pts1.shape[0]
    if pts2.shape[0] != N:
        raise ValueError("pts1 and pts2 must have the same number of points.")

    H_sym = ca.SX.sym("H_sym", 9)
    H_mat = ca.reshape(H_sym, 3, 3)
    cost = 0
    for i in range(N):
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]
        x2 = pts2[i, 0]
        y2 = pts2[i, 1]

        denom = H_mat[2, 0] * x1 + H_mat[2, 1] * y1 + H_mat[2, 2]
        x2_est = (H_mat[0, 0] * x1 + H_mat[0, 1] * y1 + H_mat[0, 2]) / denom
        y2_est = (H_mat[1, 0] * x1 + H_mat[1, 1] * y1 + H_mat[1, 2]) / denom

        cost = cost + (x2 - x2_est) ** 2 + (y2 - y2_est) ** 2

    # Define the NLP problem
    nlp = {"x": H_sym, "f": cost}

    opts = {"print_time": 0, "ipopt": {"print_level": 0}}

    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    H_init_vec = np.array(H_init, dtype=np.float64)

    x0 = H_init_vec.flatten(order="F")

    sol = solver(x0=x0)

    H_opt_vec = np.array(sol["x"]).flatten()  # 9-element vector
    H_opt = H_opt_vec.reshape((3, 3), order="F")
    H_opt = H_opt / H_opt[2, 2]

    return H_opt


def EstimateHomography(X, U):
    H = np.zeros((3, 3, X.shape[2]))
    H_casadi = np.zeros((3, 3, X.shape[2]))
    for k in range(0, X.shape[2]):

        # Solve initial Homogrpahy
        H[:, :, k] = HomographyAnalytical(X[:, :, k], U[:, :, k])

        # Find  a best solution using optimization
        H_casadi[:, :, k] = EstimateHomographyCasadi(
            X[0:2, :, k].T, U[0:2, :, k].T, H[:, :, k]
        )
    return H_casadi


def v_qp(H, p, q):
    # Extract the p-th and q-th column from H
    H_p = H[:, p]
    H_q = H[:, q]

    # Extract the elements from each column
    H_0p = H_p[0]
    H_1p = H_p[1]
    H_2p = H_p[2]

    H_0q = H_q[0]
    H_1q = H_q[1]
    H_2q = H_q[2]

    # Compute the vector V
    V = np.array(
        [
            H_0p * H_0q,
            H_0p * H_1q + H_1p * H_0q,
            H_1p * H_1q,
            H_2p * H_0q + H_0p * H_2q,
            H_2p * H_1q + H_1p * H_2q,
            H_2p * H_2q,
        ]
    )
    return V


def EstimateBMatrix(H):
    V_list = []
    n = H.shape[2]
    for k in range(n):
        Hk = H[:, :, k]
        V_12 = v_qp(Hk, 0, 1)
        V_11 = v_qp(Hk, 0, 0)
        V_22 = v_qp(Hk, 1, 1)

        V_aux = np.vstack((V_12, V_11 - V_22))

        V_list.append(V_aux)

    V = np.vstack(V_list)

    # Compute svd
    U_v, s_v, Vh_v = np.linalg.svd(V)
    b = Vh_v[-1, :]
    B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])

    # Check if B matrix is positive definte or negative definite
    try:
        L = np.linalg.cholesky(B)
        print("B is positive definite.")
    except np.linalg.LinAlgError:
        print("B is not positive definite. Checking if -B is positive definite...")
        try:
            L = np.linalg.cholesky(-B)
            print("B is negative definite.")
        except np.linalg.LinAlgError:
            print("B is neither positive nor negative definite (indefinite).")
            L = None  # or handle the indefinite case as needed

    if L is not None:
        # MATLAB: A_t = L(3,3) * (inv(L))';
        # In Python, using 0-indexing: L[2,2] * (np.linalg.inv(L)).T
        A = L[2, 2] * np.linalg.inv(L).T

        # Compute the pseudo-inverse of A.
        A_inv = np.linalg.pinv(A)
    return A


def EstimateDistortion(X, U, R_estimation, t_estimation, A):
    Identity = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    uv_c = np.array([[A[0, 2]], [A[1, 2]]])

    D_list = []
    b_list = []

    for k in range(X.shape[2]):
        U_real = U[:2, :, k]

        T_top = np.hstack([R_estimation[:, :, k], t_estimation[:, k].reshape(3, 1)])
        T_estimated = np.vstack([T_top, np.array([0, 0, 0, 1])])

        pts = np.vstack(
            (X[:2, :, k], np.zeros((1, X.shape[1])), np.ones((1, X.shape[1])))
        )

        U_estimated_hom = A @ Identity @ T_estimated @ pts
        U_estimated = U_estimated_hom[0:2, :] / U_estimated_hom[2, :]

        U_real_center = U_real - uv_c

        center_estimated = U_estimated - uv_c
        radius_estimated = np.linalg.norm(center_estimated, axis=0)

        aux_radius_square = U_real_center * (radius_estimated**2)
        aux_radius_square_square = U_real_center * (radius_estimated**4)

        aux_radius_square_reshape = aux_radius_square.reshape(
            (2 * aux_radius_square.shape[1], 1)
        )
        aux_radius_square_square_reshape = aux_radius_square_square.reshape(
            (2 * aux_radius_square_square.shape[1], 1)
        )

        aux = np.hstack([aux_radius_square_reshape, aux_radius_square_square_reshape])
        D_list.append(aux)

        # Compute the error between the real and estimated points.
        error = U_real - U_estimated
        error_reshape = error.reshape((2 * error.shape[1], 1))
        b_list.append(error_reshape)

    # After processing all samples, stack the collected arrays vertically.
    D = np.vstack(D_list)
    b = np.vstack(b_list)
    distortion = np.linalg.pinv(D) @ b

    return distortion
