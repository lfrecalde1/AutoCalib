#!/usr/bin/evn python
import numpy as np
import argparse
import matplotlib.pyplot as plt
import casadi as ca
from functions.functions import *
import time
import cv2 as cv


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
    data_uv, data_xy, objpoints, imgpoints, gray, corners_image = ComputeCorners(
        base_path, pattern_size, image_files
    )

    U = data_uv
    X = data_xy

    # Computing Homography
    H = EstimateHomography(X, U)

    # Computing intrinsic parameters
    A = EstimateBMatrix(H)

    # Computing Extrinsic parameters
    R, t = EstimatePose(H, A)

    # Computing Radial Distortion
    d = EstimateDistortion(X, U, R, t, A)

    # Set vector of initial conditions for the optimizer considering the stereographic projection
    x_init = SetInitialConditions(R, t, A, d)
    x_init = x_init.reshape((x_init.shape[0], 1))
    A_vector = np.array(
        [
            A[0, 0],
            A[0, 1],
            A[0, 2],
            A[1, 0],
            A[1, 1],
            A[1, 2],
            A[2, 0],
            A[2, 1],
            A[2, 2],
        ]
    )

    # Reshape data
    pts1_dm = X[0:2, :, 0]
    for i in range(1, X.shape[2]):
        pts1_dm = np.hstack((pts1_dm, X[0:2, :, i]))

    # Similarly for pts2:
    pts2_dm = U[0:2, :, 0]
    for i in range(1, U.shape[2]):
        pts2_dm = np.hstack((pts2_dm, U[0:2, :, i]))

    # Casadi Function to compute the cost
    f_casadi = cameraCalibrationCostFunction(2, X.shape[1], X.shape[2])
    cost = CostFunction(2, X.shape[1], X.shape[2])
    cost_classic = CostFunctionClassic(2, X.shape[1], X.shape[2])

    # Optimization problem
    solver = cameraCalibrationCasADi(X[0:2, :, :], U[0:2, :, :], f_casadi)

    # Get time to find the solution of the problem
    tic = time.time()
    sol = solver(x0=x_init)
    x_opt = sol["x"]
    x_opt = np.array(x_opt)
    toc = time.time() - tic

    # getting parameters from the optimzed vector
    A_opti, R_opti, t_opti, d_opti = GetOptiParameters(x_opt, X.shape[2])

    # print cost bafore optimization and afer it
    cost_optimization = cost(x_opt, pts1_dm, pts2_dm)
    cost_value_classic = CostFunctionMatrix(A, R, t, d, X[0:2, :, :], U[0:2, :, :])

    # checking the results of the optimizer
    # print("Rotation parameters")
    # print("Initial estimation")
    # print(R[:, :, 1])
    # print("Optimization")
    # print(R_opti[:, :, 1])

    # print("Translation parameters")
    # print("Initial estimation")
    # print(t[:, 1])
    # print("Optimization")
    # print(t_opti[:, 1])

    print("Intrinsic")
    print("Initial estimation")
    print(A)
    print("Optimization")
    print(A_opti)

    print("Distortion")
    print("Initial estimation")
    print(d)
    print("Optimization")
    print(d_opti)

    # Cost Values
    print("Cost values before optimization")
    print(cost_value_classic)
    print("Cost values after optimization")
    print(cost_optimization)

    print("Time to solve the optimization problem")
    print(toc)

    # Check the results of the estimation
    data = Results(base_path, image_files, A_opti, d_opti)
    plot_banks(data[:, 0:6], "results", base_path)
    plot_banks_detection(corners_image[:, 0:6], "results_corners", base_path)


if __name__ == "__main__":
    main()
