#python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

#import cac thu vien can thiet
import cv2
import face_recognition
import argparse
import pickle

#dinh nghia cac tham so dong lenh
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-e", "--encodings", required=True, help="path to serilaized db of facial encodings")
ap.add_argument("-d", "--detection_method", type=str, default='hog', help="face detection method to use: either 'hog' or 'cnn'")
args = vars(ap.parse_args())

#load file encoding
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
#load hinh anh
print("[INFO] loading image...")
image = cv2.imread(args["image"])
#xu li hinh anh
print("[INFO] proccessing image...")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes = face_recognition.face_locations(image, model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)
names = []
for encoding in encodings:
	matches = face_recognition.compare_faces(data["encodings"], encoding,0.4)
	name = "Unknown"
	if True in matches:
		matchesIndex = [i for (i, b) in enumerate(matches) if b]
		counts = {}
		for i in matchesIndex:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1
		name = max(counts, key=counts.get)
	names.append(name)
for((top, right, bottom, left), name) in zip(boxes, names):
	cv2.rectangle(image, (left, top), (right, bottom), (0,255,0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
