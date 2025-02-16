# AutoCalib

AutoCalib is a camera calibration tool that leverages the CasADi framework to formulate and solve a non-linear optimization problem. This project uses chessboard images to perform the calibration.

## Prerequisites

- **CasADi**  
  Install CasADi via pip:
  
  ```bash
  pip install casadi
  ```

## Setup

1. **Project Structure**  
   Ensure that your project directory contains the following:
   - A folder named `Calibration_Imgs` with your chessboard images.
   - The main script `Wrapper.py` (located in the project root).

2. **Navigate to the Project Directory**  
   Open your terminal and change to the project directory:
   
   ```bash
   cd lfrecalde_hw1
   ```

## Running the Calibration Algorithm

Execute the following command in your terminal:

```bash
python Wrapper.py
```

## Troubleshooting

- **Chessboard Images Not Found:**  
  Ensure that the images are located in the `Calibration_Imgs` folder.
  
- **CasADi Installation Issues:**  
  Confirm that CasADi is installed by running:
  
  ```bash
  pip show casadi
  ```
