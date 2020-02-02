import scipy
from scipy import ndimage
def rotateit(image, theta, isseg=False):
    order=0 if isseg==True else 5
    return scipy.ndimage.rotate(image, float(theta),reshape=False,order=order,mode='nearest')