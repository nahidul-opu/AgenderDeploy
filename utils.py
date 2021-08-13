import cv2
import skimage
import numpy as np
import os
from skimage import io as skimageIO, color
from skimage.segmentation import slic, mark_boundaries
import MRF

def AHE(img, mode="CLAHE", file_path=None):
    if mode == "HE":
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    elif mode == "CLAHE":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif mode == "skimage":
        if file_path is None:
            temp_file_path="temp.jpg"
            cv2.imwrite(temp_file_path,img)
        else:
            temp_file_path=file_path
        img = skimage.io.imread(temp_file_path)
        img = skimage.exposure.equalize_adapthist(img, clip_limit=0.03)
        img = skimage.img_as_ubyte(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if file_path is None:
            os.remove(temp_file_path)
    return img

def get_slic(img, mode="skimage",file_path=None ):
    if mode == "opencv":
        superpix = cv2.ximgproc.createSuperpixelSLIC(img, 100, 40, 10)
        superpix.iterate(300)
        labels = superpix.getLabels()
        cntr = superpix.getLabelContourMask()
    elif mode == "skimage":
        if file_path is None:
            temp_file_path="temp.jpg"
            cv2.imwrite(temp_file_path,img)
        else:
            temp_file_path=file_path
        temp_img = skimage.img_as_float(skimage.io.imread(temp_file_path))
        labels = slic(temp_img, n_segments=100, sigma=1.0)
        cntr = labels.copy()
        if file_path is None:
            os.remove(temp_file_path)
    contour = mark_boundaries(img, cntr, color=(1, 1, 1), mode="thick")
    contour = skimage.img_as_ubyte(contour)
    result = color.label2rgb(labels, img, kind="avg")
    return img, contour, result

def process_image(img,ahe=True,slic=True,mrf=True):
    ac=img.copy()
    img = AHE(img, mode="CLAHE")
    if ahe:
        _, _, img = get_slic(img, mode="opencv")
    if slic:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mrf:
        _, _, img = MRF.mrf(img,ac, max_iter=100, betha=100)
    return img