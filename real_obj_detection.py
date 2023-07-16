
import cv2
import numpy as np
import argparse
import time
import imutils
from imutils.video import VideoStream
from imutils.video import FPS

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
arg_parse.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
arg_parse.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak predictions")
args1 = vars(arg_parse.parse_args())

CLASSES_for_model= ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor","phone"]
COLORS_Random = np.random.uniform(0, 255, size=(len(CLASSES_for_model), 3))
print("[INFO] loading model...")
net_var= cv2.dnn.readNetFromCaffe(args1["prototxt"], args1["model"])
print("[INFO] starting video stream...")
video = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
while True:
	frame = video.read()
	frame = imutils.resize(frame, width=800)
	(h, w) = frame.shape[:2]
	resizing_image = cv2.resize(frame, (500, 500))
	binary_large_obj= cv2.dnn.blobFromImage(resizing_image, (1/127.5), (300, 300), 127.5, swapRB=True)
	net_var.setInput(binary_large_obj) 
	predictions = net_var.forward()
	objects = {}
	for i in np.arange(0, predictions.shape[2]):
		confidence = predictions[0, 0, i, 2]
		if confidence > args1["confidence"]:
			idx = int(predictions[0, 0, i, 1])
			box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			label = "{}: {:.2f}%".format(CLASSES_for_model[idx], confidence * 100)
			objects[CLASSES_for_model[idx]] = objects.get(CLASSES_for_model[idx], 0) + 1
			# Draw a rectangle across the boundary of the object
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS_Random[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_Random[idx], 2)
	text = ""
	for (k, v) in objects.items():
		text +="{}: {}".format(k, v)
	cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord(" "):
		break
	fps.update()
fps.stop()
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
video.stop()