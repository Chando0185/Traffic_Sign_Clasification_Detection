import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import pickle
from keras.models import load_model



frameWidth=640
frameHeight=480
brightness=100
threshold=0.90

def empty(img):
	pass

cap=cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


font=cv2.FONT_HERSHEY_COMPLEX



model = load_model('Traffic_Sign.h5')


cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar",600, 300)
cv2.createTrackbar("hue_min","TrackBar",0,179,empty)
cv2.createTrackbar("hue_max","TrackBar",179,179,empty)
cv2.createTrackbar("sat_min","TrackBar",0,255,empty)
cv2.createTrackbar("sat_max","TrackBar",255,255,empty)
cv2.createTrackbar("val_min","TrackBar",0,255,empty)
cv2.createTrackbar("val_max","TrackBar",255,255,empty)




def preprocessing(img):
    img=img.astype("uint8")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img


def get_className(classNo):
	if classNo==0:
		return "Speed limit (20km/h)"
	elif classNo==1:
		return "Speed limit (30km/h)"
	elif classNo==2:
		return "Speed limit (50km/h)"
	elif classNo==3:
		return "Speed limit (60km/h)"
	elif classNo==4:
		return "Speed limit (70km/h)"
	elif classNo==5:
		return "Speed limit (80km/h)"
	elif classNo==6:
		return "End of speed limit (80km/h)"
	elif classNo==7:
		return "Speed limit (100km/h)"
	elif classNo==8:
		return "Speed limit (120km/h)"
	elif classNo==9:
		return "No passing"
	elif classNo==10:
		return "No passing veh over 3.5 tons"
	elif classNo==11:
		return "Right-of-way at intersection"
	elif classNo==12:
		return "Priority road"
	elif classNo==13:
		return "Yield"
	elif classNo==14:
		return "Stop"
	elif classNo==15:
		return "No vehicles"
	elif classNo==16:
		return "Veh > 3.5 tons prohibited"
	elif classNo==17:
		return "No entry"
	elif classNo==18:
		return "General caution"
	elif classNo==19:
		return "Dangerous curve left"
	elif classNo==20:
		return "Dangerous curve right"
	elif classNo==21:
		return "Double curve"
	elif classNo==22:
		return "Bumpy road"
	elif classNo==23:
		return "Slippery road"
	elif classNo==24:
		return "Road narrows on the right"
	elif classNo==25:
		return "Road work"
	elif classNo==26:
		return "Traffic signals"
	elif classNo==27:
		return "Pedestrians"
	elif classNo==28:
		return "Children crossing"
	elif classNo==29:
		return "Bicycles crossing"
	elif classNo==30:
		return "Beware of ice/snow"
	elif classNo==31:
		return "Wild animals crossing"
	elif classNo==32:
		return "End speed + passing limits"
	elif classNo==33:
		return "Turn right ahead"
	elif classNo==34:
		return "Turn left ahead"
	elif classNo==35:
		return "Ahead only"
	elif classNo==36:
		return "Go straight or right"
	elif classNo==37:
		return "Go straight or left"
	elif classNo==38:
		return "Keep right"
	elif classNo==39:
		return "Keep left"
	elif classNo==40:
		return "Roundabout mandatory"
	elif classNo==41:
		return "End of no passing"
	elif classNo==42:
		return "End no passing"


while True:
	sucess, imgOrignal=cap.read()
	hsv=cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2HSV)
	img=np.asarray(imgOrignal)
	img=cv2.resize(img, (32,32))
	img=preprocessing(img)
	img=img.reshape(1, 32, 32, 1)
	cv2.putText(imgOrignal, "Class" , (20,35), font, 0.75, (0,0,255),2, cv2.LINE_AA)
	cv2.putText(imgOrignal, "Probability" , (20,75), font, 0.75, (255,0,255),2, cv2.LINE_AA)
	prediction=model.predict(img)
	classIndex=model.predict_classes(img)
	probabilityValue=np.amax(prediction)
	hue_min=cv2.getTrackbarPos("hue_min","TrackBar")
	hue_max=cv2.getTrackbarPos("hue_max","TrackBar")
	sat_min=cv2.getTrackbarPos("sat_min","TrackBar")
	sat_max=cv2.getTrackbarPos("sat_max","TrackBar")
	val_min=cv2.getTrackbarPos("val_min","TrackBar")
	val_max=cv2.getTrackbarPos("val_max","TrackBar")

	lower=np.array([hue_min, sat_min, val_min])
	upper=np.array([hue_max, sat_max, val_max])
	mask=cv2.inRange(hsv, lower, upper)
	cnts,hei=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for c in cnts:
		area=cv2.contourArea(c)
		if area>300:
			peri=cv2.arcLength(c, True)
			approx=cv2.approxPolyDP(c, 0.02*peri, True)
			x,y,w,h=cv2.boundingRect(c)
			cv2.rectangle(imgOrignal, (x,y), (x+w, y+h), (0,255,0),2)
	if probabilityValue>threshold:
		cv2.putText(imgOrignal, str(classIndex)+" "+str(get_className(classIndex)),(120,35), font, 0.75, (0,0,255),2, cv2.LINE_AA)
		cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA)
	cv2.imshow("Result",imgOrignal)
	cv2.imshow("hsv",hsv)
	cv2.imshow("Mask",mask)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()