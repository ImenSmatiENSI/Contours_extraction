from numpy.core.multiarray import concatenate
import cv2
import numpy as np
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()



def reparametrage_affine2(X, Y, N):
    # Parameterize the curve using cubic splines
    n = len(X)
    t = np.linspace(0, 1, n)

    px = CubicSpline(t, X)
    py = CubicSpline(t, Y)

    # Resample the curve with specified number of points
    X1, Y1, L = abscisse_affine2(t, px, py, N)

    return X1, Y1, L

def abscisse_affine2(t, px, py, N):
    # Compute the derivatives of the splines
    dp_x = px.derivative()
    dp2_x = dp_x.derivative()

    dp_y = py.derivative()
    dp2_y = dp_y.derivative()

    # Compute the first and second derivatives
    X1 = dp_x(t)
    X2 = dp2_x(t)
    Y1 = dp_y(t)
    Y2 = dp2_y(t)

    # Compute the integrand for arc length
    F = np.abs(X1 * Y2 - Y1 * X2)**(1/3)
    I = cumtrapz(F, t, initial=0)
    
    # Compute the normalized arc length
    if np.max(I) == 0 or not np.isfinite(np.max(I)):
        s = np.zeros_like(t)
        L = 1
    else:
        s = I / np.max(I)
        L = s[-1]

    # Resample the curve based on normalized arc length
    unique_s, index = np.unique(s, return_index=True)
    out = np.interp(np.linspace(0, 1, N), unique_s, t[index])

    # Evaluate the splines at the resampled points
    X1_val = px(out)
    Y1_val = py(out)

    return X1_val/L, Y1_val/L, L

def extract_contour2(image):
    # Create a binary image
    binary = np.zeros(image.shape, dtype=np.uint8)
    binary[image > 0] = 255

    # Apply a thresholding filter
    thresh = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY)[1]

    # Find all contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Select a contour (the first contour found)
    contour = contours[0]

    # Extract x and y coordinates of the contour
    x_coordinates = contour[:, 0, 0]
    y_coordinates = contour[:, 0, 1]

    # Check if the contour is closed
    is_closed = np.all(contour[0, 0] == contour[-1, 0])

    # Close the contour if it's not closed
    if not is_closed:
        x_coordinates = np.concatenate((x_coordinates, x_coordinates[:1],x_coordinates[:1]))
        y_coordinates = np.concatenate((y_coordinates, y_coordinates[:1],y_coordinates[:1]))

    # Resample and parameterize the contour
    x_rep, y_rep, L = reparametrage_affine2(x_coordinates,-y_coordinates,120)

    # Concatenate the x and y coordinates into two vectors
    concatenated_coordinates = np.concatenate((x_rep, y_rep))

    return concatenated_coordinates
