#!/usr/bin/evn python
import numpy as np
import cv2
import os
import argparse
from scipy.io import savemat
import matplotlib.pyplot as plt
import casadi as ca
from functions.functions import *
from scipy.spatial.transform import Rotation
import time


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


def cameraCalibrationCasADi(pts1, pts2, a_init):

    pts1 = np.array(pts1, dtype=float)
    pts2 = np.array(pts2, dtype=float)

    N = pts1.shape[2]

    n_params = 7 + 6 * N
    a_vector = ca.SX.sym("full_estimation", n_params, 1)

    cost = 0  # initialize cost

    A = ca.vertcat(
        ca.horzcat(a_vector[0], a_vector[1], a_vector[3]),
        ca.horzcat(0, a_vector[2], a_vector[4]),
        ca.horzcat(0, 0, 1),
    )

    F_identity = ca.DM.eye(3)
    Identity = ca.DM([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    vector_optimization = a_vector[7:]
    x_vector = ca.reshape(vector_optimization, 6, N)

    x = x_vector[0:3, :]
    trans = x_vector[3:6, :]

    # Loop over each sample
    for k in range(N):
        xk = x[:, k]
        norm_sq = ca.dot(xk, xk)
        denom = 1 + norm_sq
        # Map the 3-parameter vector to a quaternion (scalar-first).
        q0 = (1 - norm_sq) / denom
        q1 = 2 * xk[0] / denom
        q2 = 2 * xk[1] / denom
        q3 = 2 * xk[2] / denom
        quaternion = ca.vertcat(q0, q1, q2, q3)

        # Translation for this sample.
        trans_aux = trans[:, k]

        # REAL image coordinates (from pts2): take first two rows.
        U_real = ca.DM(pts2[0:2, :, k])

        # Compute estimated rotation matrix from quaternion.
        R_est = quat_to_rot(quaternion)
        # Build homogeneous transformation T_estimated:
        T_estimated = ca.vertcat(ca.horzcat(R_est, trans_aux), ca.DM([[0, 0, 0, 1]]))

        # Build homogeneous coordinates for pts1.
        pts1_slice = ca.DM(pts1[0:2, :, k])
        zeros_row = ca.DM.zeros(1, pts1.shape[1])
        ones_row = ca.DM.ones(1, pts1.shape[1])
        homogeneous_pts = ca.vertcat(pts1_slice, zeros_row, ones_row)

        values_normalized = F_identity @ Identity @ T_estimated @ homogeneous_pts

        aux_normalization = ca.vertcat(values_normalized[2, :], values_normalized[2, :])
        values_normalized_aux = values_normalized[0:2, :] / aux_normalization

        radius = ca.sqrt(ca.sum1(values_normalized_aux**2))
        D_expr = 1 + a_vector[5] * (radius**2) + a_vector[6] * (radius**4)
        D_aux = ca.vertcat(D_expr, D_expr)

        x_warp = values_normalized_aux * D_aux
        x_warp_aux = ca.vertcat(x_warp, ca.DM.ones(1, x_warp.size2()))

        U_improved = A @ x_warp_aux
        U_normalized_aux = ca.vertcat(U_improved[2, :], U_improved[2, :])
        U_improved_final = U_improved[0:2, :] / U_normalized_aux

        error = U_real - U_improved_final
        error_reshape = ca.reshape(error, 2 * error.size2(), 1)

        cost = cost + ca.mtimes([error_reshape.T, error_reshape])

    nlp = {"x": a_vector, "f": cost}

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
    solver = cameraCalibrationCasADi(X, U, x_init)
    tic = time.time()
    sol = solver(x0=x_init)
    x_opt = sol["x"]
    toc = time.time() - tic
    print(toc)


if __name__ == "__main__":
    main()
