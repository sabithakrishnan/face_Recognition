import cv2
import dlib
import glob
features_finalised=[]
# get the path/directory
folder_dir = '/content/drive/MyDrive/facerecognitionimages'
 
# iterate over files in
# that directory
for images in glob.iglob(f'{folder_dir}/*'):
   
    # check if the image ends with png
    if (images.endswith(".jpg")):
        img= cv2.imread(images)
        hog_face_detector = dlib.get_frontal_face_detector()
    def hogDetectFaces(image, hog_face_detector, display = True):
            height, width, _ = image.shape
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hog_face_detector(imgRGB, 0)
            for bbox in results:
                x1 = bbox.left()
                y1 = bbox.top()
                x2 = bbox.right()
                y2 = bbox.bottom()
                
            cropped_image = imgRGB[x1:x2, y1:y2]
            

            return cropped_image
    output=hogDetectFaces(img, hog_face_detector, display=True)
    grayimg = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(grayimg)
    import numpy as np
    import skimage.feature as feature
    graycom = feature.graycomatrix(equ, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    contrast = feature.graycoprops(graycom, 'contrast')
    dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
    homogeneity = feature.graycoprops(graycom, 'homogeneity')
    energy = feature.graycoprops(graycom, 'energy')
    correlation = feature.graycoprops(graycom, 'correlation')
    ASM = feature.graycoprops(graycom, 'ASM')
    features=[max(max(contrast)), max(max(dissimilarity)), max(max(homogeneity)), max(max(energy)), max(max(correlation)), max(max(ASM))]
    features_finalised.append(features)
x=features_finalised
y=[1,1,1,1,1,0,0,0,0,0]
from sklearn.svm import SVC  
clf = SVC(C=1000, gamma=5) 
  
# fitting x samples and y classes 
clf.fit(x,y) 
import pickle
model_pkl_file = "facetrainingmodel.pkl"  
with open(model_pkl_file, 'wb') as file:  
    pickle.dump(clf, file)



