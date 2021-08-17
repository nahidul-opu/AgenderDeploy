import numpy as np
import cv2
import pickle
import os
from skimage.feature import hog
from mahotas.features import surf,zernike,haralick,lbp,pftas
import joblib
from sklearn.cluster import KMeans
bins = 8
from sklearn.cluster import MiniBatchKMeans

def ulbp(image):
    radius = 2
    n_points = 12
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Features = lbp(gray, radius, n_points)
    return Features

def discrete_cosine_transform(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image)/255.0
    
    image = cv2.resize(image,(32,32))
    dct = cv2.dct(image)
    
    dct = cv2.resize(dct,(8,8))
    dct = cv2.dct(dct)
    
    dct = np.uint8(dct*255.0)
    dct = np.array(dct)
    dct = dct.flatten()
    avg=np.average(dct)
    dct.reshape(-1)
    x = np.where(dct<np.average(dct),0,1)
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


def HSV_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def RGB_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def zernike_moments(image):
    radius = 15
    degree = 8
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Features = zernike(gray,radius,degree)
    return Features

def tas(image):
    features = pftas(image)
    return features

def feature_hog(image):
    img = image[:, :, ::-1]
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True,feature_vector=True)
    return fd

class SIFT:
    def __init__(self,df,k=1000):
        dico = []
        self.k=k
        self.sift = cv2.SIFT_create()
        for row in df.iterrows():
            img_path = row[1]["image_dir"].replace("flowers", "Augmented")
            t = os.path.splitext(img_path)
            for name in ["","_zoom","_hshift","_vshift","_flip",'_rotate']:
                p = t[0]+name+t[1]
                img = cv2.imread(img_path)
                kp, des = self.sift.detectAndCompute(img, None)
                for d in des:
                    dico.append(d)
                
        self.kmeans = MiniBatchKMeans(n_clusters=self.k).fit(dico)
        with open("output/sift.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)
    
    def feature(self,img):
        kp, des = self.sift.detectAndCompute(img, None)
        histo = np.zeros(self.k)
        nkp = np.size(kp)
        for d in des:
            idx = self.kmeans.predict([d])
            histo[idx] += 1/nkp
        return histo
    
class ORB:
    def __init__(self,df,k=1000):
        dico = []
        self.k=k
        self.orb = cv2.ORB_create()
        for row in df.iterrows():
            img_path = row[1]["image_dir"].replace("flowers", "Augmented")
            t = os.path.splitext(img_path)
            for name in ["","_zoom","_hshift","_vshift","_flip",'_rotate']:
                p = t[0]+name+t[1]
                img = cv2.imread(img_path)
                kp, des = self.orb.detectAndCompute(img, None)
                for d in des:
                    dico.append(d)
                
        self.kmeans = MiniBatchKMeans(n_clusters=self.k).fit(dico)
        with open("output/orb.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)
    
    def feature(self,img):
        kp, des = self.orb.detectAndCompute(img, None)
        histo = np.zeros(self.k)
        nkp = np.size(kp)
        for d in des:
            idx = self.kmeans.predict([d])
            histo[idx] += 1/nkp
        return histo
    
class SURF:
    def __init__(self,df,k=1000):
        dico = []
        self.k=k
        for row in df.iterrows():
            img_path = row[1]["image_dir"].replace("flowers", "Augmented")
            t = os.path.splitext(img_path)
            for name in ["","_zoom","_hshift","_vshift","_flip",'_rotate']:
                p = t[0]+name+t[1]
                img = cv2.imread(img_path)
                img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                des = surf.surf(img, descriptor_only = True)
                for d in des:
                    dico.append(d)
                
        self.kmeans = MiniBatchKMeans(n_clusters=self.k).fit(dico)
        with open("output/surf.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)
    
    def feature(self,img):
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        des = surf.surf(img, descriptor_only = True)
        histo = np.zeros(self.k)
        nkp = np.size(des)
        for d in des:
            idx = self.kmeans.predict([d])
            histo[idx] += 1/nkp
        return histo
    
def feature_sift(img,kmeans,k=1000):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    histo = np.zeros(k)
    nkp = np.size(kp)
    for d in des:
        idx = kmeans.predict([d])
        histo[idx] += 1/nkp
    return histo

def feature_orb(img,kmeans,k=1000):
    with open("output/orb.pkl", "rb") as f:
        kmeans = pickle.load(f)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    histo = np.zeros(k)
    nkp = np.size(kp)
    for d in des:
        idx = kmeans.predict([d])
        histo[idx] += 1/nkp
    return histo

def feature_surf(img,kmeans,k=1000):
    with open("output/surf.pkl", "rb") as f:
        kmeans = pickle.load(f)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    des = surf.surf(img, descriptor_only = True)
    histo = np.zeros(k)
    nkp = np.size(des)
    for d in des:
        idx = kmeans.predict([d])
        histo[idx] += 1/nkp
    return histo

def color_moments(img):
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average 
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

    return color_feature

def extract_features(image):
    
    with open("output/sift.pkl", "rb") as f:
        sift = pickle.load(f)
    with open("output/orb.pkl", "rb") as f:
        orb = pickle.load(f)
    with open("output/surf.pkl", "rb") as f:
        surf = pickle.load(f)
    global_features = []
    #featuresLBP = ulbp(image)
    featuresZernike = zernike_moments(image)
    featuresHaralick = haralick_texture(image)
    featuresRGBHistogram = RGB_histogram(image)
    #featuresHSVHistogram = HSV_histogram(image)
    featuresMoments = hu_moments(image)
    #featuresTAS = tas(image)
    featuresSIFT = feature_sift(image,sift)
    featuresSURF = feature_orb(image,orb)
    featuresORB = feature_surf(image,surf)
    #featuresCTM = color_moments(image)
    global_feature = np.hstack(
        [
           #featuresLBP,
           featuresZernike,
           featuresHaralick,
           featuresRGBHistogram,
           #featuresHSVHistogram,
           featuresMoments,
           #featuresTAS,
           featuresSIFT,
           featuresSURF,
           featuresORB,
           #featuresCTM
        ]
    )
    global_features.append(global_feature)
    scaler = joblib.load("output/Scaler.save") 
    rescaled_features = scaler.transform(global_features)
    return np.array(rescaled_features)