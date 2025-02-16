#!/usr/bin/evn python
import numpy as np
import cv2
import os
from scipy.io import savemat
import matplotlib.pyplot as plt
import casadi as ca
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as mpatches


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


def Results(base_path, image_files, camera_matrix, dist_coeffs):
    # Loop to read the images
    empty_matrix_pb = np.empty((2, len(image_files)), dtype=object)
    for k, image_name in enumerate(image_files):
        full_file_name = os.path.join(base_path, image_name)
        print(f"Reading and processing image: {full_file_name}")

        I = cv2.imread(full_file_name)
        if I is None:
            print(f"Failed to load image: {full_file_name}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        dst = cv2.undistort(gray, camera_matrix, dist_coeffs, None, newcameramtx)

        corners = cv2.goodFeaturesToTrack(
            dst, maxCorners=100, qualityLevel=0.01, minDistance=10
        )

        # If corners were found, draw them on the image
        if corners is not None:
            # Convert corner coordinates to integers
            corners = np.int32(corners)
            for corner in corners:
                x_corner, y_corner = corner.ravel()
                cv2.circle(
                    dst,
                    (x_corner, y_corner),
                    radius=3,
                    color=(255, 255, 255),
                    thickness=20,
                )

        # crop the image
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]

        empty_matrix_pb[0, k] = gray
        empty_matrix_pb[1, k] = dst

    return empty_matrix_pb


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

    # Aux values to compare with baseline
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    empty_matrix_pb = np.empty((1, len(image_files)), dtype=object)

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
            drawn_image = gray.copy()
            cv2.drawChessboardCorners(drawn_image, pattern_size, corners, ret)

            image_points = corners.reshape(-1, 2).T
            u_points = image_points[0, :]
            v_points = image_points[1, :]

            # Order the points of the cheesboard in oder to have points as the ones I defined in the model points
            u_points = u_points.reshape((ny, nx))
            v_points = v_points.reshape((ny, nx))
            u_points = u_points[:, ::-1]
            v_points = v_points[:, ::-1]
            u_points = u_points.reshape((1, nx * ny))
            v_points = v_points.reshape((1, nx * ny))

            # Save points Image Plane
            data_uv[0, :, k] = u_points
            data_uv[1, :, k] = v_points

            # Save points model plane
            data_xy[0:2, :, k] = points2D[0:2, :]

            objpoints.append(objp)
            imgpoints.append(corners)

            empty_matrix_pb[0, k] = drawn_image
        else:
            print(f"Checkerboard not detected in image: {full_file_name}")

    savemat(
        "calibration_values_python.mat",
        {"data_uv_python": data_uv, "data_xy_python": data_xy},
    )
    print("Calibration data saved to calibration_values.mat")
    return data_uv, data_xy, objpoints, imgpoints, gray, empty_matrix_pb


def Normalization(X):
    x_mean = np.mean(X[0, :])
    y_mean = np.mean(X[1, :])

    # Compute the sample variances for x and y
    x_var = np.var(X[0, :], ddof=1)
    y_var = np.var(X[1, :], ddof=1)

    # Compute scaling factors
    s_x = np.sqrt(2.0 / x_var)
    s_y = np.sqrt(2.0 / y_var)

    # Construct the normalization matrix
    H_norm = np.array([[s_x, 0, -s_x * x_mean], [0, s_y, -s_y * y_mean], [0, 0, 1]])

    return H_norm


def EstimatePose(H, A):
    # Create empty variables for the rotation matrices
    R = np.zeros((3, 3, H.shape[2]))

    # Translation vectors
    t = np.zeros((3, H.shape[2]))
    A_inv = np.linalg.pinv(A)

    for k in range(H.shape[2]):
        # Split columns of the Homography matrix
        h_0 = H[:, 0, k]
        h_1 = H[:, 1, k]
        h_2 = H[:, 2, k]

        # Compute scaling factor
        lambda_aux = 1 / np.linalg.norm(A_inv @ h_0)

        # Compute rotation matrix
        r_0 = lambda_aux * (A_inv @ h_0)
        r_1 = lambda_aux * (A_inv @ h_1)
        r_2 = np.cross(r_0, r_1)

        # Create rotation matrix by stacking the columns computed before
        R_aux = np.column_stack((r_0, r_1, r_2))

        # Refining the rotation matrix to guarantee orthogonality
        U_r, _, Vh_r = np.linalg.svd(R_aux)
        R[:, :, k] = U_r @ Vh_r  # Note: Vh_r is V transposed

        # Compute the translation vector
        t[:, k] = lambda_aux * (A_inv @ h_2)

    return R, t


def HomographyAnalytical(X, U):

    # Normalization of the matrices
    H_X = Normalization(X[0:2, :])
    H_U = Normalization(U[0:2, :])

    # Transform the points into normalized coordinates
    X_norm = H_X @ X
    U_norm = H_U @ U

    # Empty matrix
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

    # Compute SVD to solve Ax = 0
    _, _, Vh = np.linalg.svd(A)
    h_nomr = Vh[-1, :]

    # A solution that minimizes the norm
    H_norm = h_nomr.reshape((3, 3))

    # Inverse normalization to obtain the original homography matrix
    H = np.linalg.pinv(H_U) @ H_norm @ H_X
    return H


def EstimateHomographyCasadi(pts1, pts2, H_init):
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    # Number of poinst of the model plane
    N = pts1.shape[0]

    # check for the same number of points in the image plane
    if pts2.shape[0] != N:
        raise ValueError("pts1 and pts2 must have the same number of points.")

    # Create symbolic variables
    H_sym = ca.SX.sym("H_sym", 9)
    H_mat = ca.reshape(H_sym, 3, 3)

    # Reshape the symbolic optimization variables in order to have a matrix

    # Init cost value
    cost = 0

    # Cost over the different views(Images)
    for i in range(N):
        # Extracting points of the model
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]
        # Extracting points of the  image plane
        x2 = pts2[i, 0]
        y2 = pts2[i, 1]

        # Computing projectin over the image plane
        denom = H_mat[2, 0] * x1 + H_mat[2, 1] * y1 + H_mat[2, 2]
        x2_est = (H_mat[0, 0] * x1 + H_mat[0, 1] * y1 + H_mat[0, 2]) / denom
        y2_est = (H_mat[1, 0] * x1 + H_mat[1, 1] * y1 + H_mat[1, 2]) / denom

        # Compute cost
        cost = cost + (x2 - x2_est) ** 2 + (y2 - y2_est) ** 2

    # Define the NLP problem for casadi
    nlp = {"x": H_sym, "f": cost}

    opts = {"print_time": 0, "ipopt": {"print_level": 0}}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # Defines the init Homography for the problem
    H_init_vec = np.array(H_init, dtype=np.float64)
    x0 = H_init_vec.flatten(order="F")

    # Solves the problem
    sol = solver(x0=x0)
    H_opt_vec = np.array(sol["x"]).flatten()  # 9-element vector

    # Normalize the solution
    H_opt = H_opt_vec.reshape((3, 3), order="F")
    H_opt = H_opt / H_opt[2, 2]
    return H_opt


def EstimateHomography(X, U):
    # Create empty matrix to store the homography matrices
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
        A = L[2, 2] * np.linalg.inv(L).T
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

        error = U_real - U_estimated
        error_reshape = error.reshape((2 * error.shape[1], 1))
        b_list.append(error_reshape)

    D = np.vstack(D_list)
    b = np.vstack(b_list)

    # Solution for the distortion parameters
    distortion = np.linalg.pinv(D) @ b
    return distortion


def GetOptiParameters(a_vector, samples):
    N = samples
    # Intrinsic parameters
    A = np.array(
        [
            [a_vector[0, 0], a_vector[1, 0], a_vector[3, 0]],
            [0.0, a_vector[2, 0], a_vector[4, 0]],
            [0, 0, 1],
        ]
    )

    # Optimization variables related to the orientation and position
    vector_optimization = a_vector[7:]
    x_vector = ca.reshape(vector_optimization, 6, N)

    x = x_vector[0:3, :]  # quaternions in a vector space
    trans = x_vector[3:6, :]  # translations

    R = np.zeros((3, 3, N))
    t = np.zeros((3, N))
    d = np.array([a_vector[5, 0], a_vector[6, 0], 0.0, 0.0, 0.0])

    for k in range(N):
        q_0 = (1 - x[0:3, k].T @ x[0:3, k]) / (1 + x[0:3, k].T @ x[0:3, k])
        q_1 = 2 * x[0, k] / (1 + x[0:3, k].T @ x[0:3, k])
        q_2 = 2 * x[1, k] / (1 + x[0:3, k].T @ x[0:3, k])
        q_3 = 2 * x[2, k] / (1 + x[0:3, k].T @ x[0:3, k])

        quaternion = np.array([float(q_0), float(q_1), float(q_2), float(q_3)])

        t[:, k] = np.array([float(trans[0, k]), float(trans[1, k]), float(trans[2, k])])
        R[:, :, k] = quat_to_rot(quaternion)

    return A, R, t, d


def CostFunctionClassic(n, points, samples):
    # Dimensions
    n_rows = n
    n_cols = points
    N = samples

    # Number of data points total
    n_pts_total = n_cols * N

    # Numebr of optimization variables hard code values
    n_params = 7 + 6 * N

    # Define symbolic variables (where we have the parameters to optimize and the inputs such as points in the model plane and image plane)
    a_vector = ca.SX.sym("full_estimation", n_params, 1)
    A_aux = ca.SX.sym("A_aux", 9, 1)
    pts1_sym = ca.SX.sym("pts1", n_rows, n_pts_total)
    pts2_sym = ca.SX.sym("pts2", n_rows, n_pts_total)

    # Intrinsic parameters
    A = ca.vertcat(
        ca.horzcat(A_aux[0], A_aux[1], A_aux[2]),
        ca.horzcat(A_aux[3], A_aux[4], A_aux[5]),
        ca.horzcat(A_aux[6], A_aux[7], A_aux[8]),
    )

    F_identity = ca.DM.eye(3)
    Identity = ca.DM([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    const_transform = F_identity @ Identity

    # Precompute
    zeros_row = ca.DM.zeros(1, n_cols)
    ones_row = ca.DM.ones(1, n_cols)
    aux_last_element_homogeneous = ca.DM([[0, 0, 0, 1]])

    # Optimization variables related to the orientation and position
    vector_optimization = a_vector[7:]
    x_vector = ca.reshape(vector_optimization, 6, N)
    x = x_vector[0:3, :]  # quaternions in a vector space
    trans = x_vector[3:6, :]  # translations

    cost = 0
    for k in range(N):
        # Determine the column slice for the k-th sample (data stored as 2 x (points*N)).
        start_index = k * n_cols
        end_index = (k + 1) * n_cols

        # Getting the values of the model plane and the image plane
        pts1_slice = pts1_sym[:, start_index:end_index]  # shape: (n_rows x n_cols)
        pts2_slice = pts2_sym[:, start_index:end_index]  # shape: (n_rows x n_cols)
        U_real = pts2_slice

        # Mapping the Euclidean space to quaternions
        xk = x[:, k]  # 3x1 vector
        norm_sq = ca.dot(xk, xk)
        denom = 1 + norm_sq
        q0 = (1 - norm_sq) / denom
        q1 = 2 * xk[0] / denom
        q2 = 2 * xk[1] / denom
        q3 = 2 * xk[2] / denom
        quaternion = ca.vertcat(q0, q1, q2, q3)

        trans_aux = trans[:, k]

        # Computing the rotation matrix and translation
        R_est = quat_to_rot(quaternion)
        T_estimated = ca.vertcat(
            ca.horzcat(R_est, trans_aux), aux_last_element_homogeneous
        )

        # Create homogeneous coordinates for pts1_slice
        homogeneous_pts = ca.vertcat(pts1_slice, zeros_row, ones_row)
        values_normalized = const_transform @ T_estimated @ homogeneous_pts
        aux_normalization = ca.vertcat(values_normalized[2, :], values_normalized[2, :])

        # Normalized values
        values_normalized_aux = values_normalized[0:2, :] / aux_normalization

        # Computing Radial Distortion
        radius = ca.sqrt(ca.sum1(values_normalized_aux**2))
        D_expr = 1 + a_vector[5] * (radius**2) + a_vector[6] * (radius**4)
        D_aux = ca.vertcat(D_expr, D_expr)
        x_warp = values_normalized_aux * D_aux

        # Create homogeneous coordinates for the warped points
        x_warp_aux = ca.vertcat(x_warp, ones_row)
        U_improved = A @ x_warp_aux
        U_normalized_aux = ca.vertcat(U_improved[2, :], U_improved[2, :])
        U_improved_final = U_improved[0:2, :] / U_normalized_aux

        # Compute error M view
        error = U_real - U_improved_final  # shape: (n_rows x n_cols)
        cost = cost + ca.norm_fro(error)

    # everthing in a vector
    total_cost = cost / N

    # Create the CasADi function.
    f_cost = ca.Function(
        "f_cost",
        [a_vector, A_aux, pts1_sym, pts2_sym],
        [total_cost],
        ["a_vector", "A_aux", "pts1", "pts2"],
        ["cost"],
    )
    return f_cost


def CostFunction(n, points, samples):
    # Dimensions
    n_rows = n
    n_cols = points
    N = samples

    # Number of data points total
    n_pts_total = n_cols * N

    # Numebr of optimization variables hard code values
    n_params = 7 + 6 * N

    # Define symbolic variables (where we have the parameters to optimize and the inputs such as points in the model plane and image plane)
    a_vector = ca.SX.sym("full_estimation", n_params, 1)
    pts1_sym = ca.SX.sym("pts1", n_rows, n_pts_total)
    pts2_sym = ca.SX.sym("pts2", n_rows, n_pts_total)

    # Intrinsic parameters
    A = ca.vertcat(
        ca.horzcat(a_vector[0], 0.0, a_vector[3]),
        ca.horzcat(0, a_vector[2], a_vector[4]),
        ca.horzcat(0, 0, 1),
    )

    F_identity = ca.DM.eye(3)
    Identity = ca.DM([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    const_transform = F_identity @ Identity

    # Precompute
    zeros_row = ca.DM.zeros(1, n_cols)
    ones_row = ca.DM.ones(1, n_cols)
    aux_last_element_homogeneous = ca.DM([[0, 0, 0, 1]])

    # Optimization variables related to the orientation and position
    vector_optimization = a_vector[7:]
    x_vector = ca.reshape(vector_optimization, 6, N)
    x = x_vector[0:3, :]  # quaternions in a vector space
    trans = x_vector[3:6, :]  # translations

    cost = 0
    for k in range(N):
        # Determine the column slice for the k-th sample (data stored as 2 x (points*N)).
        start_index = k * n_cols
        end_index = (k + 1) * n_cols

        # Getting the values of the model plane and the image plane
        pts1_slice = pts1_sym[:, start_index:end_index]  # shape: (n_rows x n_cols)
        pts2_slice = pts2_sym[:, start_index:end_index]  # shape: (n_rows x n_cols)
        U_real = pts2_slice

        # Mapping the Euclidean space to quaternions
        xk = x[:, k]  # 3x1 vector
        norm_sq = ca.dot(xk, xk)
        denom = 1 + norm_sq
        q0 = (1 - norm_sq) / denom
        q1 = 2 * xk[0] / denom
        q2 = 2 * xk[1] / denom
        q3 = 2 * xk[2] / denom
        quaternion = ca.vertcat(q0, q1, q2, q3)

        trans_aux = trans[:, k]

        # Computing the rotation matrix and translation
        R_est = quat_to_rot(quaternion)
        T_estimated = ca.vertcat(
            ca.horzcat(R_est, trans_aux), aux_last_element_homogeneous
        )

        # Create homogeneous coordinates for pts1_slice
        homogeneous_pts = ca.vertcat(pts1_slice, zeros_row, ones_row)
        values_normalized = const_transform @ T_estimated @ homogeneous_pts
        aux_normalization = ca.vertcat(values_normalized[2, :], values_normalized[2, :])

        # Normalized values
        values_normalized_aux = values_normalized[0:2, :] / aux_normalization

        # Computing Radial Distortion
        radius = ca.sqrt(ca.sum1(values_normalized_aux**2))
        D_expr = 1 + a_vector[5] * (radius**2) + a_vector[6] * (radius**4)
        D_aux = ca.vertcat(D_expr, D_expr)
        x_warp = values_normalized_aux * D_aux

        # Create homogeneous coordinates for the warped points
        x_warp_aux = ca.vertcat(x_warp, ones_row)
        U_improved = A @ x_warp_aux
        U_normalized_aux = ca.vertcat(U_improved[2, :], U_improved[2, :])
        U_improved_final = U_improved[0:2, :] / U_normalized_aux

        # Compute error M view
        error = U_real - U_improved_final  # shape: (n_rows x n_cols)
        cost = cost + ca.norm_fro(error)

    # everthing in a vector
    total_cost = cost / N

    # Create the CasADi function.
    f_cost = ca.Function(
        "f_cost",
        [a_vector, pts1_sym, pts2_sym],
        [total_cost],
        ["a_vector", "pts1", "pts2"],
        ["cost"],
    )
    return f_cost


def CostFunctionMatrix(A, R, t, d, pts1, pts2):
    # Dimensions
    n_rows = pts1.shape[0]
    n_cols = pts1.shape[1]
    N = pts1.shape[2]

    F_identity = np.eye(3)
    Identity = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    const_transform = F_identity @ Identity

    # Precompute
    zeros_row = np.zeros((1, n_cols))
    ones_row = np.ones((1, n_cols))
    aux_last_element_homogeneous = np.array([[0, 0, 0, 1]])
    cost = 0

    for k in range(N):
        # Getting the values of the model plane and the image plane
        U_real = pts2[:, :, k]
        trans_aux = t[:, k]

        # Computing the rotation matrix and translation
        R_est = R[:, :, k]
        T_estimated = np.hstack((R_est, trans_aux.reshape((3, 1))))
        T_estimated = np.vstack((T_estimated, aux_last_element_homogeneous))

        # Create homogeneous coordinates for pts1_slice
        homogeneous_pts = np.vstack((pts1[:, :, k], zeros_row, ones_row))
        values_normalized = const_transform @ T_estimated @ homogeneous_pts
        aux_normalization = np.vstack(
            (values_normalized[2, :], values_normalized[2, :])
        )

        # Normalized values
        values_normalized_aux = values_normalized[0:2, :] / values_normalized[2, :]

        # Computing Radial Distortion
        radius = np.linalg.norm(values_normalized_aux, axis=0)
        D_expr = 1 + d[0] * (radius**2) + d[1] * (radius**4)
        D_aux = np.vstack((D_expr, D_expr))
        x_warp = values_normalized_aux * D_expr

        # Create homogeneous coordinates for the warped points
        x_warp_aux = np.vstack((x_warp, ones_row))
        U_improved = A @ x_warp_aux
        U_normalized_aux = np.vstack((U_improved[2, :], U_improved[2, :]))
        U_improved_final = U_improved[0:2, :] / U_improved[2, :]

        # Compute error M view
        error = U_real - U_improved_final  # shape: (n_rows x n_cols)
        cost = cost + np.linalg.norm(error, "fro")

    # everthing in a vector
    total_cost = cost / N

    return total_cost


def cameraCalibrationCostFunction(n, points, samples):
    # Dimensions
    n_rows = n
    n_cols = points
    N = samples

    # Number of data points total
    n_pts_total = n_cols * N

    # Numebr of optimization variables hard code values
    n_params = 7 + 6 * N

    # Define symbolic variables (where we have the parameters to optimize and the inputs such as points in the model plane and image plane)
    a_vector = ca.SX.sym("full_estimation", n_params, 1)
    pts1_sym = ca.SX.sym("pts1", n_rows, n_pts_total)
    pts2_sym = ca.SX.sym("pts2", n_rows, n_pts_total)

    # Intrinsic parameters
    A = ca.vertcat(
        ca.horzcat(a_vector[0], 0.0, a_vector[3]),
        ca.horzcat(0, a_vector[2], a_vector[4]),
        ca.horzcat(0, 0, 1),
    )

    F_identity = ca.DM.eye(3)
    Identity = ca.DM([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    const_transform = F_identity @ Identity

    # Precompute
    zeros_row = ca.DM.zeros(1, n_cols)
    ones_row = ca.DM.ones(1, n_cols)
    aux_last_element_homogeneous = ca.DM([[0, 0, 0, 1]])

    # Optimization variables related to the orientation and position
    vector_optimization = a_vector[7:]
    x_vector = ca.reshape(vector_optimization, 6, N)
    x = x_vector[0:3, :]  # quaternions in a vector space
    trans = x_vector[3:6, :]  # translations

    error_blocks = []
    for k in range(N):
        # Determine the column slice for the k-th sample (data stored as 2 x (points*N)).
        start_index = k * n_cols
        end_index = (k + 1) * n_cols

        # Getting the values of the model plane and the image plane
        pts1_slice = pts1_sym[:, start_index:end_index]  # shape: (n_rows x n_cols)
        pts2_slice = pts2_sym[:, start_index:end_index]  # shape: (n_rows x n_cols)
        U_real = pts2_slice

        # Mapping the Euclidean space to quaternions
        xk = x[:, k]  # 3x1 vector
        norm_sq = ca.dot(xk, xk)
        denom = 1 + norm_sq
        q0 = (1 - norm_sq) / denom
        q1 = 2 * xk[0] / denom
        q2 = 2 * xk[1] / denom
        q3 = 2 * xk[2] / denom
        quaternion = ca.vertcat(q0, q1, q2, q3)

        trans_aux = trans[:, k]

        # Computing the rotation matrix and translation
        R_est = quat_to_rot(quaternion)
        T_estimated = ca.vertcat(
            ca.horzcat(R_est, trans_aux), aux_last_element_homogeneous
        )

        # Create homogeneous coordinates for pts1_slice
        homogeneous_pts = ca.vertcat(pts1_slice, zeros_row, ones_row)
        values_normalized = const_transform @ T_estimated @ homogeneous_pts
        aux_normalization = ca.vertcat(values_normalized[2, :], values_normalized[2, :])

        # Normalized values
        values_normalized_aux = values_normalized[0:2, :] / aux_normalization

        # Computing Radial Distortion
        radius = ca.sqrt(ca.sum1(values_normalized_aux**2))
        D_expr = 1 + a_vector[5] * (radius**2) + a_vector[6] * (radius**4)
        D_aux = ca.vertcat(D_expr, D_expr)
        x_warp = values_normalized_aux * D_aux

        # Create homogeneous coordinates for the warped points
        x_warp_aux = ca.vertcat(x_warp, ones_row)
        U_improved = A @ x_warp_aux
        U_normalized_aux = ca.vertcat(U_improved[2, :], U_improved[2, :])
        U_improved_final = U_improved[0:2, :] / U_normalized_aux

        # Compute error M view
        error = U_real - U_improved_final  # shape: (n_rows x n_cols)
        error_vec = ca.reshape(error, n_rows * n_cols, 1)
        error_blocks.append(error_vec)

    # everthing in a vector
    all_errors = ca.vertcat(*error_blocks)
    total_cost = ca.dot(all_errors, all_errors)

    # Create the CasADi function.
    f_cost = ca.Function(
        "f_cost",
        [a_vector, pts1_sym, pts2_sym],
        [total_cost],
        ["a_vector", "pts1", "pts2"],
        ["cost"],
    )
    return f_cost


def quat_to_rot(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    R = ca.vertcat(
        ca.horzcat(
            1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)
        ),
        ca.horzcat(
            2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)
        ),
        ca.horzcat(
            2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)
        ),
    )
    return R


def cameraCalibrationCasADi(pts1, pts2, f_casadi):

    pts1 = np.array(pts1, dtype=float)
    pts2 = np.array(pts2, dtype=float)

    # Number of views
    N = pts1.shape[2]

    n_params = 7 + 6 * N
    a_vector = ca.SX.sym("full_estimation", n_params, 1)
    # Reshape elements model plane so we have the data in a huge matrix
    pts1_dm = pts1[:, :, 0]
    for i in range(1, pts1.shape[2]):
        pts1_dm = np.hstack((pts1_dm, pts1[:, :, i]))

    # Reshape elements image plane so we have the data in a huge matrix
    pts2_dm = pts2[:, :, 0]
    for i in range(1, pts2.shape[2]):
        pts2_dm = np.hstack((pts2_dm, pts2[:, :, i]))

    # Compute cost directly from my function
    result = f_casadi(a_vector, pts1_dm, pts2_dm)
    cost = result
    nlp = {"x": a_vector, "f": cost}

    H_cost = ca.hessian(cost, a_vector)[0]

    # Get the sparsity pattern.
    H_sparsity = H_cost.sparsity()

    # Visualize the sparsity pattern.
    plt.figure(figsize=(6, 6))
    plt.spy(H_sparsity, markersize=3)
    plt.title("Sparsity Pattern of the Calibration Cost Hessian")
    plt.xlabel("Opt Variables")
    plt.ylabel("Opt Variables")
    # plt.show()

    # IPOPT options.
    opts = {"print_time": True, "ipopt": {"print_level": 5}}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    return solver


def SetInitialConditions(R_estimation, t_estimation, A, d):
    # Reshape my rotation matrix in order to have matrices compatible with the library
    R_aux = np.zeros((R_estimation.shape[2], 3, 3))
    for k in range(R_estimation.shape[2]):
        R_aux[k, :, :] = R_estimation[:, :, k]

    # R_matrices = np.transpose(R_estimation, (2, 0, 1))
    rotations = Rotation.from_matrix(R_aux)
    quat_scipy = rotations.as_quat()  # shape: (n_samples, 4)

    # Quaternions are in the following form w, x, y, z
    quaternion_estimated = np.column_stack(
        (quat_scipy[:, 3], quat_scipy[:, 0], quat_scipy[:, 1], quat_scipy[:, 2])
    )
    negative_mask = quaternion_estimated[:, 0] < 0
    quaternion_estimated[negative_mask] *= -1
    distortion_1 = d[0, 0]
    distortion_2 = d[1, 0]
    # Initial solution from the intrinsic parameters
    X_init_list = [
        A[0, 0],
        A[0, 1],
        A[1, 1],
        A[0, 2],
        A[1, 2],
        float(distortion_1),
        float(distortion_2),
    ]

    # Loop through each sample
    for k in range(R_estimation.shape[2]):
        x_quaternion = quaternion_estimated[k, 1:4] / quaternion_estimated[k, 0]
        X_init_list.extend(x_quaternion.tolist() + t_estimation[:, k].tolist())

    X_init = np.array(X_init_list)
    return X_init


def cameraCalibrationOpti(n, points, samples, pts1_dm, pts2_dm):
    n_rows = n  # e.g., 2
    n_cols = points  # e.g., 54
    N = samples  # e.g., 13
    n_pts_total = n_cols * N  # e.g., 702
    n_params = 7 + 6 * N  # global (7) + per-sample (6 each)

    opti = ca.Opti()

    # Define the optimization variable
    a_vector = opti.variable(n_params, 1)
    dummy = opti.variable()

    # Define constant parameters (they are fixed and known)
    pts1 = opti.parameter(n_rows, n_pts_total)
    pts2 = opti.parameter(n_rows, n_pts_total)
    opti.set_value(pts1, pts1_dm)
    opti.set_value(pts2, pts2_dm)

    # Define the intrinsic matrix A (global parameters in a_vector[0:5]).
    A = ca.vertcat(
        ca.horzcat(a_vector[0], a_vector[1], a_vector[3]),
        ca.horzcat(0, a_vector[2], a_vector[4]),
        ca.horzcat(0, 0, 1),
    )

    # Precompute constant matrices.
    F_identity = ca.DM.eye(3)
    Identity = ca.DM([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    const_transform = F_identity @ Identity

    zeros_row = ca.DM.zeros(1, n_cols)
    ones_row = ca.DM.ones(1, n_cols)

    # The remaining parameters (starting at index 7) are per-sample.
    vector_optimization = a_vector[7:]
    x_vector = ca.reshape(vector_optimization, 6, N)
    x = x_vector[0:3, :]  # shape: 3 x N (for rotation)
    trans = x_vector[3:6, :]  # shape: 3 x N (for translation)

    # Build the cost as a sum of squared errors from each sample.
    error_blocks = []
    for k in range(N):
        start_index = k * n_cols
        end_index = (k + 1) * n_cols

        pts1_slice = pts1[:, start_index:end_index]
        pts2_slice = pts2[:, start_index:end_index]

        # Compute quaternion from 3 parameters (for sample k).
        xk = x[:, k]
        norm_sq = ca.dot(xk, xk)
        denom = 1 + norm_sq
        q0 = (1 - norm_sq) / denom
        q1 = 2 * xk[0] / denom
        q2 = 2 * xk[1] / denom
        q3 = 2 * xk[2] / denom
        quaternion = ca.vertcat(q0, q1, q2, q3)

        # Transformation.
        trans_aux = trans[:, k]
        R_est = quat_to_rot(quaternion)
        T_estimated = ca.vertcat(ca.horzcat(R_est, trans_aux), ca.DM([[0, 0, 0, 1]]))

        # Warp pts1.
        homogeneous_pts = ca.vertcat(pts1_slice, zeros_row, ones_row)
        values_normalized = const_transform @ T_estimated @ homogeneous_pts
        aux_normalization = ca.vertcat(values_normalized[2, :], values_normalized[2, :])
        values_normalized_aux = values_normalized[0:2, :] / aux_normalization

        # Apply distortion.
        radius = ca.sqrt(ca.sum1(values_normalized_aux**2))
        D_expr = 1 + a_vector[5] * (radius**2) + a_vector[6] * (radius**4)
        D_aux = ca.vertcat(D_expr, D_expr)
        x_warp = values_normalized_aux * D_aux

        x_warp_aux = ca.vertcat(x_warp, ca.DM.ones(1, n_cols))
        U_improved = A @ x_warp_aux
        U_normalized_aux = ca.vertcat(U_improved[2, :], U_improved[2, :])
        U_improved_final = U_improved[0:2, :] / U_normalized_aux

        error = pts2_slice - U_improved_final
        error_vec = ca.reshape(error, n_rows * n_cols, 1)
        error_blocks.append(error_vec)

    # Stack error blocks and define the total cost.
    all_errors = ca.vertcat(*error_blocks)
    total_cost = ca.dot(all_errors, all_errors)
    opti.minimize(total_cost)
    opti.subject_to(dummy == 0)

    # Set IPOPT options.
    opti.solver("ipopt", {"expand": True})

    return opti, a_vector


def plot_banks(bank, name, location, include_classical=True):
    n, m = bank.shape

    # Create a figure with subplots
    fig, axs = plt.subplots(n, m, figsize=(1.5 * m, 2 * n))
    axs = np.ravel(axs)

    # Loop over each image in the bank
    for i in range(n):
        for j in range(m):
            subplot_idx = i * m + j
            ax = axs[subplot_idx]

            # Plot the image in grayscale
            im = ax.imshow(bank[i, j], cmap="gray", interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])

            # If this subplot is in the first column, set its y-axis label.
            if j == 0:
                if i == 0:
                    ax.set_ylabel("Original Image", fontsize=8)
                elif i == 1:
                    ax.set_ylabel("Undistort Image", fontsize=8)

    # Add a colorbar to the figure
    fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)

    # Save the figure as a PDF
    file_path = os.path.join(location, f"{name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    plt.clf()
    plt.close()

    print(f"PDF with colorbar saved to: {file_path}")

    return None


def plot_banks_detection(bank, name, location, include_classical=True):
    n, m = bank.shape

    # Create a figure with subplots
    fig, axs = plt.subplots(n, m, figsize=(1.5 * m, 2 * n))
    axs = np.ravel(axs)

    # Loop over each image in the bank
    for i in range(n):
        for j in range(m):
            subplot_idx = i * m + j
            ax = axs[subplot_idx]

            # Plot the image in grayscale
            im = ax.imshow(bank[i, j], cmap="gray", interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])

            # If this subplot is in the first column, set its y-axis label.
            if j == 0:
                if i == 0:
                    ax.set_ylabel("Original Image", fontsize=8)

    # Add a colorbar to the figure
    fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)

    # Save the figure as a PDF
    file_path = os.path.join(location, f"{name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    plt.clf()
    plt.close()

    print(f"PDF with colorbar saved to: {file_path}")

    return None
