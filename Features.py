import numpy as np
import cv2
from skimage.feature import hog
from mahotas.features import surf,zernike,haralick,lbp,pftas
import joblib
from sklearn.cluster import KMeans
bins = 8

def ulbp(image):
    radius = 2
    n_points = 12
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Features = lbp(gray, radius, n_points)
    return Features

def discrete_cosine_transform(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image)/255.0
    stepSize = 16
    windowSize = (32, 32)
    '''dct = []
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            current = image[y : y + windowSize[1], x : x + windowSize[0]]
            dct_current = cv2.dct(image)
            dct_current = np.sort(dct_current,axis=None)[:32]
            dct = dct + list(dct_current)'''
    dct = cv2.dct(image)
    dct = np.uint8(dct*255.0)
    dct = np.array(dct)
    dct = dct.flatten()
    dct.reshape(-1)
    #dct = np.sort(dct,axis=None)[:int((image.shape[0]*image.shape[0])/2)]
    return dct

def hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def haralick_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = haralick(gray).mean(axis=0)
    return feature


def color_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def zernike_moments(image):
    radius = 15
    degree = 8
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Features = zernike(gray,radius,degree)
    return Features

def feature_surf(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    textures = surf.surf(np.asanyarray(gray))
    Features = textures.mean(axis=0)
    return Features

def tas(image):
    features = pftas(image)
    return features

def feature_sift(image):
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    kmn = KMeans(n_clusters=1)
    kmn.fit(des)
    return kmn.cluster_centers_.flatten()

def feature_hog(image):
    img = image[:, :, ::-1]
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True,feature_vector=True)
    return fd

def extract_features(image):
    global_features = []
    featuresLBP = ulbp(image)
    featuresZernike = zernike_moments(image)
    featuresHaralick = haralick_texture(image)
    featuresHistogram = color_histogram(image)
    featuresMoments = hu_moments(image)
    featuresSurf = feature_surf(image)
    featuresTAS = tas(image)

    global_feature = np.hstack(
        [
            featuresLBP,
            featuresZernike,
            featuresHaralick,
            featuresHistogram,
            featuresMoments,
            featuresSurf,
            featuresTAS,
        ]
    )
    global_features.append(global_feature)
    scaler = joblib.load("output/Scaler.save") 
    rescaled_features = scaler.transform(global_features)
    return np.array(rescaled_features)