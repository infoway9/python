import face_recognition
import numpy as np
import cv2
from keras.preprocessing import image

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
sumanta_image = face_recognition.load_image_file("sumanta.jpg")
sumanta_face_encoding = face_recognition.face_encodings(sumanta_image)[0]

# Load a second sample picture and learn how to recognize it.
anubrata_image = face_recognition.load_image_file("anubrata.jpg")
anubrata_face_encoding = face_recognition.face_encodings(anubrata_image)[0]

# Load a second sample picture and learn how to recognize it.
subhadeepsheet_image = face_recognition.load_image_file("subhadeepsheet.jpg")
subhadeepsheet_face_encoding = face_recognition.face_encodings(subhadeepsheet_image)[0]

piyal_image = face_recognition.load_image_file("piyal.jpg")
piyal_face_encoding = face_recognition.face_encodings(piyal_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    sumanta_face_encoding,
    anubrata_face_encoding,
    subhadeepsheet_face_encoding,
    piyal_face_encoding
]
known_face_names = [
    "Sumanta",
    "Anubrata",
    "Subhadeep",
    "Piyal"
]

#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights

#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

while(True):
	ret, img = cap.read()
	#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

	rgb_frame = img[:, :, ::-1]

	# Find all the faces and face enqcodings in the frame of video
	face_locations = face_recognition.face_locations(rgb_frame)
	face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	#print(faces) #locations of detected faces

	for (x,y,w,h), face_encoding in zip(face_locations, face_encodings):
    		
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
		
		detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
		predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
		max_index = np.argmax(predictions[0])
		
		emotion = emotions[max_index]
		
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
		name = "Unknown Face"
		face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
		best_match_index = np.argmin(face_distances)
		if matches[best_match_index]:
    			name = known_face_names[best_match_index]

		#write emotion text above rectangle
		cv2.putText(img, name + " - " + emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		
		#process on detected face end
		#-------------------------

	cv2.imshow('img',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()