### Simple Image processing tools for Borst lab
import cv2


import cv2
import numpy as np
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
