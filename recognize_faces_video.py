# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/video_output.avi --display 0
#import cac thu vien can thiet
import cv2
import face_recognition
import imutils
import time
import argparse
import pickle
from imutils.video import VideoStream
#dinh nghia cac doi so dong lenh
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection_method", type=str, default='hog', help="face detection method to use: either 'hog' or 'cnn'")
ap.add_argument("-o", "--output", type=str, help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1, help="where or not to dispaly output frame on screen")
args = vars(ap.parse_args())

#load file encoding
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
#khoi dong camera
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(3.0)
while True:
	frame = vs.read()
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(rgb, width=680, height=480)
	r = frame.shape[1]/float(rgb.shape[1])
	boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"], encoding, 0.4)
		name = "Unknown"
		if True in matches:
			matchesIndex = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			for i in matchesIndex:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			name = max(counts, key=counts.get)
		names.append(name)
	for ((top, right, bottom, left), name) in zip(boxes, names):
		top = int(top*r)
		right = int(right*r)
		bottom = int(bottom*r)
		left = int(left*r)
		cv2.rectangle(frame, (left, top),(right, bottom), (0,255,0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 10, (frame.shape[1], frame.shape[0]), True)
	if writer is not None:
		writer.write(frame)
	if args["display"]>0:
		cv2.imshow("Display Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
cv2.destroyAllWindows()
vs.stop()
if writer is not None:
		writer.release()
