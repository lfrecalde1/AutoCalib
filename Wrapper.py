#!/usr/bin/evn python
import numpy as np
import argparse
import matplotlib.pyplot as plt
import casadi as ca
from functions.functions import *
from scipy.spatial.transform import Rotation
import time


def cameraCalibrationCostFunction(n, points, samples):
    # Dimensions
    n_rows = n  # e.g., 2
    n_cols = points  # e.g., 54 points per sample
    N = samples  # e.g., 13 samples
    n_pts_total = n_cols * N  # e.g., 702 columns
    n_params = 7 + 6 * N  # first 7 are global, then 6 per sample

    # Define symbolic variables.
    a_vector = ca.SX.sym("full_estimation", n_params, 1)
    pts1_sym = ca.SX.sym("pts1", n_rows, n_pts_total)
    pts2_sym = ca.SX.sym("pts2", n_rows, n_pts_total)

    # Define the intrinsic matrix A (global parameters appear in a_vector[0:5]).
    A = ca.vertcat(
        ca.horzcat(a_vector[0], a_vector[1], a_vector[3]),
        ca.horzcat(0, a_vector[2], a_vector[4]),
        ca.horzcat(0, 0, 1),
    )

    # Precompute constant matrices (these do not depend on the optimization variables).
    F_identity = ca.DM.eye(3)
    Identity = ca.DM([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    const_transform = F_identity @ Identity

    # Precompute rows used for creating homogeneous coordinates.
    zeros_row = ca.DM.zeros(1, n_cols)
    ones_row = ca.DM.ones(1, n_cols)

    # The remaining parameters (starting at index 7) are per–sample.
    vector_optimization = a_vector[7:]
    x_vector = ca.reshape(vector_optimization, 6, N)
    # For each sample, first three parameters yield a quaternion (rotation) and the next three the translation.
    x = x_vector[0:3, :]  # shape: (3 x N)
    trans = x_vector[3:6, :]  # shape: (3 x N)

    # Instead of summing the cost term-by-term, we build a list of error blocks.
    error_blocks = []
    for k in range(N):
        # Determine the column slice for the k-th sample (data stored as 2 x (points*N)).
        start_index = k * n_cols
        end_index = (k + 1) * n_cols
        pts1_slice = pts1_sym[:, start_index:end_index]  # shape: (n_rows x n_cols)
        pts2_slice = pts2_sym[:, start_index:end_index]  # shape: (n_rows x n_cols)

        # --- Compute the quaternion from the 3 parameters in x for sample k ---
        xk = x[:, k]  # 3x1 vector
        norm_sq = ca.dot(xk, xk)
        denom = 1 + norm_sq
        q0 = (1 - norm_sq) / denom
        q1 = 2 * xk[0] / denom
        q2 = 2 * xk[1] / denom
        q3 = 2 * xk[2] / denom
        quaternion = ca.vertcat(q0, q1, q2, q3)

        # --- Transformation for this sample ---
        trans_aux = trans[:, k]
        # U_real: real image coordinates for sample k (from pts2).
        U_real = pts2_slice

        # Compute the estimated rotation matrix from the quaternion.
        R_est = quat_to_rot(quaternion)
        T_estimated = ca.vertcat(ca.horzcat(R_est, trans_aux), ca.DM([[0, 0, 0, 1]]))

        # --- Warp pts1 ---
        # Create homogeneous coordinates for pts1_slice.
        homogeneous_pts = ca.vertcat(pts1_slice, zeros_row, ones_row)
        values_normalized = const_transform @ T_estimated @ homogeneous_pts
        aux_normalization = ca.vertcat(values_normalized[2, :], values_normalized[2, :])
        values_normalized_aux = values_normalized[0:2, :] / aux_normalization

        # --- Apply distortion ---
        radius = ca.sqrt(ca.sum1(values_normalized_aux**2))
        D_expr = 1 + a_vector[5] * (radius**2) + a_vector[6] * (radius**4)
        D_aux = ca.vertcat(D_expr, D_expr)
        x_warp = values_normalized_aux * D_aux

        # Create homogeneous coordinates for the warped points.
        x_warp_aux = ca.vertcat(x_warp, ca.DM.ones(1, n_cols))
        U_improved = A @ x_warp_aux
        U_normalized_aux = ca.vertcat(U_improved[2, :], U_improved[2, :])
        U_improved_final = U_improved[0:2, :] / U_normalized_aux

        # --- Compute error for sample k ---
        error = U_real - U_improved_final  # shape: (n_rows x n_cols)
        # Flatten the error to a column vector.
        error_vec = ca.reshape(error, n_rows * n_cols, 1)
        # Append to the list: each block only depends on global and sample k’s parameters.
        error_blocks.append(error_vec)

    # Stack all error blocks vertically.
    all_errors = ca.vertcat(*error_blocks)
    # Total cost: sum of squared errors.
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
    # q is a 4x1 CasADi expression with elements [q0, q1, q2, q3]
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

    N = pts1.shape[2]

    n_params = 7 + 6 * N
    a_vector = ca.SX.sym("full_estimation", n_params, 1)
    pts1_dm = pts1[:, :, 0]  # Start with the first slice, shape: (2, 54)
    for i in range(1, pts1.shape[2]):
        pts1_dm = np.hstack((pts1_dm, pts1[:, :, i]))

    # Similarly for pts2:
    pts2_dm = pts2[:, :, 0]
    for i in range(1, pts2.shape[2]):
        pts2_dm = np.hstack((pts2_dm, pts2[:, :, i]))

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
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()

    # IPOPT options.
    opts = {"print_time": True, "ipopt": {"print_level": 5}}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    return solver


def SetInitialConditions(R_estimation, t_estimation, A):
    n_samples = R_estimation.shape[2]
    R_matrices = np.transpose(R_estimation, (2, 0, 1))
    rotations = Rotation.from_matrix(R_matrices)
    quat_scipy = rotations.as_quat()  # shape: (n_samples, 4)

    quaternion_estimated = np.column_stack(
        (quat_scipy[:, 3], quat_scipy[:, 0], quat_scipy[:, 1], quat_scipy[:, 2])
    )

    X_init_list = [A[0, 0], A[0, 1], A[1, 1], A[0, 2], A[1, 2], 0, 0]

    # Loop through each sample
    for k in range(R_estimation.shape[2]):
        x_quaternion = quaternion_estimated[k, 1:] / quaternion_estimated[k, 0]

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
    p_opts = {"expand": True}
    s_opts = {"print_level": 0, "max_iter": 1000}
    opti.solver("ipopt", {"expand": True})

    return opti, a_vector


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="./Calibration_Imgs/",
        help="Base path of images, Default:/Images",
    )

    Parser.add_argument(
        "--patternSize", default="7,10", help="Checkerboard pattern size default '7,10'"
    )

    # Set up parameters
    args = Parser.parse_args()
    base_path = args.BasePath

    try:
        pattern_size = tuple(map(int, args.patternSize.split(",")))
    except Exception as e:
        print(
            "Error parsing patternSize. Please provide two integers separated by a comma (e.g., '7,7')."
        )
        return

    # Get images names
    image_files = GetImagesNames(base_path)

    # Compute Corners and point over the cheesboard plane
    data_uv, data_xy = ComputeCorners(base_path, pattern_size, image_files)

    U = data_uv
    X = data_xy
    H = EstimateHomography(X, U)
    A = EstimateBMatrix((H))
    R, t = EstimatePose(H, A)
    d = EstimateDistortion(X, U, R, t, A)
    x_init = SetInitialConditions(R, t, A)

    # Reshape data
    pts1_dm = X[0:2, :, 0]  # Start with the first slice, shape: (2, 54)
    for i in range(1, X.shape[2]):
        pts1_dm = np.hstack((pts1_dm, X[0:2, :, i]))

    # Similarly for pts2:
    pts2_dm = U[0:2, :, 0]
    for i in range(1, U.shape[2]):
        pts2_dm = np.hstack((pts2_dm, U[0:2, :, i]))

    # Create Optimization
    opti, a_vector = cameraCalibrationOpti(2, X.shape[1], X.shape[2], pts1_dm, pts2_dm)
    opti.set_initial(a_vector, x_init)
    tic = time.time()
    sol = opti.solve()
    toc_libray = time.time() - tic
    print("Solution found in", toc_libray, "seconds")

    f_casadi = cameraCalibrationCostFunction(2, X.shape[1], X.shape[2])
    solver = cameraCalibrationCasADi(X[0:2, :, :], U[0:2, :, :], f_casadi)
    tic = time.time()
    sol = solver(x0=x_init)
    x_opt = sol["x"]
    toc = time.time() - tic
    print(toc, toc_libray)


if __name__ == "__main__":
    main()
