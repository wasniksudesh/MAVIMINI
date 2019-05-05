# USAGE
# python ncs_realtime_objectdetection.py --graph graphs/mobilenetgraph --display 1
# python ncs_realtime_objectdetection.py --graph graphs/mobilenetgraph --confidence 0.5 --display 1

# import the necessary packages
from mvnc import mvncapi as mvnc
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import numpy as np
import time
import cv2
import pyttsx3

# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ("background", "aeroplane", "bicycle", "bird",
	"boat", "bottle", "bus", "car", "cat", "chair", "cow",
	"diningtable", "dog", "horse", "motorbike", "person",
	"pottedplant", "sheep", "sofa", "train", "tvmonitor")


# frame dimensions should be sqaure
PREPROCESS_DIMS = (300, 300)

engine=pyttsx3.init()

def preprocess_image(input_image):
	# preprocess the image
	preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
	preprocessed = preprocessed - 127.5
	preprocessed = preprocessed * 0.007843
	preprocessed = preprocessed.astype(np.float16)

	# return the image to the calling function
	return preprocessed

def predict(image, graph):
	# preprocess the image
	image = preprocess_image(image)

	# send the image to the NCS and run a forward pass to grab the
	# network predictions
	graph.LoadTensor(image, None)
	(output, _) = graph.GetResult()

	# grab the number of valid object predictions from the output,
	# then initialize the list of predictions
	num_valid_boxes = output[0].astype(int)
	predictions = []

	# loop over results
	for box_index in range(num_valid_boxes):
		# calculate the base index into our array so we can extract
		# bounding box information
		base_index = 7 + box_index * 7

		# boxes with non-finite (inf, nan, etc) numbers must be ignored
		if (not np.isfinite(output[base_index]) or
			not np.isfinite(output[base_index + 1]) or
			not np.isfinite(output[base_index + 2]) or
			not np.isfinite(output[base_index + 3]) or
			not np.isfinite(output[base_index + 4]) or
			not np.isfinite(output[base_index + 5]) or
			not np.isfinite(output[base_index + 6])):
			continue

		# extract the image width and height and clip the boxes to the
		# image size in case network returns boxes outside of the image
		# boundaries
		(h, w) = image.shape[:2]
		x1 = max(0, int(output[base_index + 3] * w))
		y1 = max(0, int(output[base_index + 4] * h))
		x2 = min(w,	int(output[base_index + 5] * w))
		y2 = min(h,	int(output[base_index + 6] * h))

		# grab the prediction class label, confidence (i.e., probability),
		# and bounding box (x, y)-coordinates
		pred_class = int(output[base_index + 1])
		pred_conf = output[base_index + 2]
		pred_boxpts = ((x1, y1), (x2, y2))

		# create prediciton tuple and append the prediction to the
		# predictions list
		prediction = (pred_class, pred_conf, pred_boxpts)
		predictions.append(prediction)

	# return the list of predictions to the calling function
	return predictions

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--graph", required=True,
	help="path to input graph file")
ap.add_argument("-c", "--confidence", default=.5,
	help="confidence threshold")
ap.add_argument("-d", "--display", type=int, default=0,
	help="switch to display image on screen")
args = vars(ap.parse_args())

# grab a list of all NCS devices plugged in to USB
print("[INFO] finding NCS devices...")
devices = mvnc.EnumerateDevices()

# if no devices found, exit the script
if len(devices) == 0:
	print("[INFO] No devices found. Please plug in a NCS")
	quit()

# use the first device since this is a simple test script
# (you'll want to modify this is using multiple NCS devices)
print("[INFO] found {} devices. device0 will be used. "
	"opening device0...".format(len(devices)))
device = mvnc.Device(devices[0])
device.OpenDevice()

# open the CNN graph file
print("[INFO] loading the graph file into RPi memory...")
with open(args["graph"], mode="rb") as f:
	graph_in_memory = f.read()

# load the graph into the NCS
print("[INFO] allocating the graph on the NCS...")
graph = device.AllocateGraph(graph_in_memory)

# open a pointer to the video stream thread and allow the buffer to
# start to fill, then start the FPS counter
print("[INFO] starting the video stream and FPS counter...")
#"Please put the destination of the video here in place of 'df.mp4'"
vs = cv2.VideoCapture('df.mp4')

time.sleep(1)
fps = FPS().start()

# loop over frames from the video file stream
while(vs.isOpened()):
	try:
		# grab the frame from the threaded video stream
		# make a copy of the frame and resize it for display/video purposes
		ret,frame = vs.read()
		if(cv2.waitKey(1) & 0xFF ==ord('q')):
			break;	
		
		# use the NCS to acquire predictions
		predictions = predict(frame, graph)

		# loop over our predictions
		x=0;
		c=0;
		a=0;
		for (i, pred) in enumerate(predictions):
			# extract prediction data for readability
			(pred_class, pred_conf, pred_boxpts) = pred

			# filter out weak detections by ensuring the `confidence`
			# is greater than the minimum confidence
			if pred_conf > args["confidence"]:
				#print("clss is ", CLASSES[pred_class])
				if(pred_conf>c):
					x=pred_class;
					c=pred_conf;
					a=pred_boxpts[0][0]
		l="center"
		if(x!=0):
			if(a<100):
				l="left"
			elif(a>200):
				l="right"
			print(CLASSES[x]);
			engine.say(CLASSES[x])
			engine.say(l);
		engine.runAndWait();
		print("AScAS")
		time.sleep(1)


		# update the FPS counter
		fps.update()
	
	# if "ctrl+c" is pressed in the terminal, break from the loop
	except KeyboardInterrupt:
		break

	# if there's a problem reading a frame, break gracefully
	except AttributeError:
		break

# stop the FPS counter timer
fps.stop()

# destroy all windows if we are displaying them
if args["display"] > 0:
	cv2.destroyAllWindows()

# stop the video stream
vs.stop()

# clean up the graph and device
graph.DeallocateGraph()
device.CloseDevice()

# display FPS information
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
