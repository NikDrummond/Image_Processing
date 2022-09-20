### Simple Image processing tools for Borst lab

import cv2
from scipy.spatial import ConvexHull
from scipy.ndimage.filters import gaussian_filter

from .Thresholding import *
from .Image_Class import Image,ImageList
from .Image_Metrics import get_pixels

# contrast enhancement

def contrast_enhancement(N, clipLimit = 2.0, tileGridSize = (8,8)):

    """
    CLAHE
    """
    # convert to uint8
    N.array = N.array.astype(np.uint8)
    #create CLAHE obj and apply change
    clahe = cv2.createCLAHE(clipLimit = clipLimit,tileGridSize = tileGridSize)
    N.array = clahe.apply(N.array)

    # convert back to float 32
    N.array = N.array.astype(np.float32)

    return N

def get_points(N,t = 0,order = 'x,y',inplace = True):
    points = np.argwhere(N.array > t)

    if order == 'x,y':
        points = points[:,[1,0]]
    elif order == 'y,x':
        pass
    else:
        raise TypeError('input order string not recognised')
        
    if inplace == True:
        N.points = points
        return N
    else:
        return points

def add_hull(N,t = 0, order = 'x,y'):
    """
    fit a Convex hull to an image
    """

    if N.points is None:
        N = get_points(N,t = t,order = order)
    N.hull = ConvexHull(N.points)
    N.hull.density = get_pixels(N, t = t) / N.hull.volume
    return N

def threshold(N, t = None, apply = True, inplace = True, **kwargs):
    """
    Threshold an image using Generalised Histogram Thresholding    
    
    """
    if not isinstance(N, Image):
        if isinstance(N, ImageList):
            raise TypeError('Please specify single image from your list')
        else:
            raise TypeError('Input type not recognised')

    im = N.array

    if t is None:

        n,x = image_hist(im)
        t = GHT(n,x,**kwargs)[0]

        if apply == False:
            return t
        else:
            im = N.array.copy()

    mask = im > t
    mask = np.array(mask).astype(bool)
    im[mask == False] = 0

    if inplace == False:
        return im
    else:
        N.array = im
        return N

def blur_image(N, sigma = 8, inplace = True):
    """
    Gaussian blurring of an image
    """
    im = N.array.copy()
    im = gaussian_filter(im,sigma = sigma)

    if inplace == False:
        return im
    else:
        N.array = im
        return N

def extract_mask(N,t=0):
    """
    Generate a masking array to be applied to an image
    """
    mask = N.array.copy()
    mask = mask > t
    mask = np.array(mask).astype(bool)
    return mask

def apply_mask(N,mask,invert = False,inplace = True):
    """ 
    Given a boolian mask, apply it to an image
    
    """
    im = N.array.copy()
    im[mask == invert] = 0

    if inplace == False:
        return im
    else:
        N.array = im
        return N
