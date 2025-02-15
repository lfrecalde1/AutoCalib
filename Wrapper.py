#!/usr/bin/evn python
import numpy as np
import argparse
import matplotlib.pyplot as plt
import casadi as ca
from functions.functions import *
from scipy.spatial.transform import Rotation
import time


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
