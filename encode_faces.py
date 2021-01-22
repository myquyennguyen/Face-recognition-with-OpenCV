# python encode_faces.py --dataset dataset --encodings encodings.pickle
#import cac thu vien can thiet
import cv2
import face_recognition
import os
import pickle
import argparse
from imutils import paths
#dinh nghia cac doi so dong lenh
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces+images")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection_method", type=str, default='hog',help="face detection method to use: either 'hog' or 'cnn'")
args = vars(ap.parse_args())

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"])) #lay duong dan den cac anh trong thu muc
knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] proccessing image {}/{}".format(i+1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	
	image = cv2.resize(src=image,dsize=None, fx=0.4,fy=0.4)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	#rgb = cv2.resize(src=rgb,dsize=(800,900))
	
	boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"],"wb")
f.write(pickle.dumps(data))
f.close()

