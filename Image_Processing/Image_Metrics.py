
import math

import cv2
import numpy as np
import pandas as pd
from numpy import array

from .Image_Class import Image, ImageList


def get_pixels(N, t= 0):
    """
    count number of non zero pixels
    """
    bool_mat = N.array > t
    return bool_mat.sum()

def get_volume(N):
    """
    Get area of neuron(in pixels)
    """
    return N.hull.volume

def get_density(N):
    """
    get the density of the Neuron
    """
    return N.hull.density

def rotate(origin, point, angle, direction = 'counterclockwise'):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    if direction == 'counterclockwise':
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    elif direction == 'clockwise':
        qx = ox + math.cos(angle) * (px - ox) + math.sin(angle) * (py - oy)
        qy = oy - math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy

def angle_between(p1,p2):
    """
    angle beteen 2 points
    """
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1-ang2 % (2 * np.pi)))

def image_PCA(coords):
    """
    x/y coordinates for each PCA of a 2D set of coordinates
    """
    cov_mat = np.cov(coords)
    evals,evecs = np.linalg.eig(cov_mat)

    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  
    x_v2, y_v2 = evecs[:, sort_indices[1]]

    return [x_v1,y_v1,x_v2,y_v2]

def rebase_coordinates(coords):
    """
    Rebase an images coordinate system to align with it's principle cmoponents
    """
    
    x_v1,y_v1,x_v2,y_v2 = image_PCA(coords)
    theta = np.arctan((x_v1)/(y_v1))
    rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),np.cos(theta)]])

    rotated_coords = rotation_mat * coords
    x_transformed, y_transformed = rotated_coords.A

    return x_transformed, y_transformed

def get_image_coords(image,origin = None):
    """
    Returns the image coordinates as an array of two vectors. If origin is not none, the the given origin will be subrtacted from each axis. If origin is a scalar, the value will be subtracted from all axis
    """
    # get coords of pixels above threshold
    coords = np.array(np.where(image.array >0))

    if origin is not None:

        if len(origin) > 1:
            # subtract entry point and do some organising
            coords[1] = coords[1] - origin[0]
            coords[0] = coords[0] - origin[1]
        else:
            coords[1] = coords[1] - origin
            coords[0] = coords[0] - origin


    coords = np.vstack((coords[1],coords[0]))

    return coords

def get_directionality(x,y):
    """
    Basic measure of magnitude along each axis
    """
    # the fraction of pixels in each direction
    signed_x = np.sign(x)
    signed_y = np.sign(y)
    frac_x_pos = len(signed_x[signed_x == 1]) /len(x)
    frac_x_neg = 1 - frac_x_pos
    frac_y_pos = len(signed_y[signed_y == 1]) /len(y)
    frac_y_neg = 1 - frac_y_pos

    # The difference
    diff_x = abs(frac_x_pos - frac_x_neg)
    diff_y = abs(frac_y_pos - frac_y_neg)

    # then the difference times the fraction in that direction
    x_pos = diff_x * frac_x_pos
    x_neg = diff_x * frac_x_neg
    y_pos = diff_y * frac_y_pos
    y_neg = diff_y * frac_y_neg

    return x_neg,x_pos,y_neg,y_pos

def get_asymmetry(x,y,origin = None):
    """
    Simple Measure of Asymmetry
    """
    # if origin is not [0,0], shift x and y
    if origin is not None:
        if len(origin) > 1:
            # subtract entry point and do some organising
            x = x - origin[0]
            y = y - origin[1]
        else:
            x = x - origin
            y = y - origin
    # the fraction of pixels in each direction
    signed_x = np.sign(x)
    signed_y = np.sign(y)
    frac_x_pos = len(signed_x[signed_x == 1]) /len(x)
    frac_x_neg = 1 - frac_x_pos
    frac_y_pos = len(signed_y[signed_y == 1]) /len(y)
    frac_y_neg = 1 - frac_y_pos

    # The difference
    diff_x = abs(frac_x_pos - frac_x_neg)
    diff_y = abs(frac_y_pos - frac_y_neg)

    return (diff_x + diff_y)/2

#everything as one...
def get_directional_data(image,x,y):
    """

    """

    # get image coordinates
    coords = get_image_coords(image,origin = [x,y])
    # rotate to align with the PCAs
    x_transformed,y_transformed = rebase_coordinates(coords)

    # get fractional metric
    x_neg,x_pos,y_neg,y_pos = get_directionality(x_transformed,y_transformed)

    # we want the minimum and maximum values along each axis
    x_max = np.max(x_transformed)
    x_min = np.min(x_transformed)
    y_max = np.max(y_transformed)
    y_min = np.min(y_transformed)

    # and finally, multiply by the scalar value from the first point. 
    final_x_pos = x_max * x_pos
    final_x_neg = x_min * x_neg
    final_y_pos = y_max * y_pos
    final_y_neg = y_min * y_neg

    # define angle/theta
    x_v1,y_v1,x_v2,y_v2 = image_PCA(coords)
    theta = np.arctan((x_v1)/(y_v1))

    ## sort out the vectors - we have the points, and the origin is [0,0], so rotate the second point by -theta degrees

    x_pos_final = rotate(origin = [0,0], point = [final_x_pos,0], angle = theta, direction = 'clockwise')
    x_neg_final = rotate(origin = [0,0], point = [final_x_neg,0], angle = theta, direction = 'clockwise')
    y_pos_final = rotate(origin = [0,0], point = [0,final_y_pos], angle = theta, direction = 'clockwise')
    y_neg_final = rotate(origin = [0,0], point = [0,final_y_neg], angle = theta, direction = 'clockwise')

    # get angle of each vector
    x_pos_angle = angle_between([0,0],x_pos_final)
    x_neg_angle = angle_between([0,0], x_neg_final)
    y_pos_angle = angle_between([0,0], y_pos_final)
    y_neg_angle = angle_between([0,0], y_neg_final)

    df = pd.DataFrame.from_dict({'axis':['x_positive','x_negative','y_positive','y_negative'],
                                'Fraction_weight':[x_pos,x_neg,y_pos,y_neg],
                                'Pixel_scale':[x_max,x_min,y_max,y_min],
                                'Angle':[x_pos_angle,x_neg_angle,y_pos_angle,y_neg_angle],
                                'xy': [x_pos_final,x_neg_final,y_pos_final,y_neg_final]})

    return df


