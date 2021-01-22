import cv2
import os
import time
import argparse
import imutils
from imutils.video import VideoStream
import face_recognition
import imutils
import time
import argparse
import pickle


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detection_method", type=str, default='hog', help="face detection method to use: either 'hog' or 'cnn'")
ap.add_argument("-o", "--output", required=True,  help="path to output dicrection")
args = vars(ap.parse_args())


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(3.0)
s=0
total=0
while True:
	frame = vs.read()
	frame2 = frame.copy()
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(rgb, width=680, height=480)
	r = frame.shape[1]/float(rgb.shape[1])
	boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
	for (top, right, bottom, left) in boxes:
		top = int(top*r)
		right = int(right*r)
		bottom = int(bottom*r)
		left = int(left*r)
		cv2.rectangle(frame, (left, top),(right, bottom), (0,255,0), 2)
	cv2.imshow("Display Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("k"):
		p = os.path.sep.join([args["output"], "{}.png".format(str(s).zfill(5))])
		cv2.imwrite(p, frame2)
		print("{} face image stored...".format(s+1))
		s+=1
	elif key == ord("q"):
		break
print("[INFO] clearning up...")
cv2.destroyAllwindows()
vs.stop()

