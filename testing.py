import cv2
import pickle
import dlib
import matplotlib.pyplot as plt
files=open("/content/facetrainingmodel.pkl",'rb')
model = pickle.load(files)
hog_face_detector = dlib.get_frontal_face_detector()
def hogDetectFaces(image, hog_face_detector, display = True):

            height, width, _ = image.shape

            output_image = image.copy()

            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hog_face_detector(imgRGB, 0)

            for bbox in results:

                x1 = bbox.left()
                y1 = bbox.top()
                x2 = bbox.right()
                y2 = bbox.bottom()
                cv2.rectangle(output_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=width//200)  
                plt.figure(figsize=[15,15])
                plt.subplot(131);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
                plt.subplot(132);plt.imshow(output_image[:,:,::-1]);plt.title("detected face");plt.axis('off');
            cropped_image = imgRGB[x1:x2, y1:y2]
            plt.subplot(133);plt.imshow(cropped_image);plt.title("Extracted face");plt.axis('off');



            return cropped_image
image = cv2.imread('/content/drive/MyDrive/facerecognitionimages/Natasha_Lyonne_0001.jpg')

output=hogDetectFaces(image, hog_face_detector, display=True)
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
op=model.predict([features])
print(op)
plt.figure()
plt.imshow(image[:,:,::-1])
if op==1:
  plt.title('RECOGNISED')
else:
  plt.title('UNRECOGNISED')
 