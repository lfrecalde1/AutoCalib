#!/usr/bin/evn python
import numpy as np
import cv2
import os
import argparse
from scipy.io import savemat
import matplotlib.pyplot as plt


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
    print(data_uv[:, 1, 1])
    print(data_xy[:, 1, 1])


if __name__ == "__main__":
    main()
