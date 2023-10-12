import cv2
#load the haar-cascade classifier files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
#load the input image
img = cv2.imread('/Users/j/Documents/image.jpeg')
cv2.imshow('Original Image',img)
cv2.waitKey(0)
#convert input image to grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#apply the detection methods on the grayscale image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
   cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
   roi_gray = gray[y:y+h, x:x+w]
   roi_color = img[y:y+h, x:x+w]
eyes = eye_cascade.detectMultiScale(roi_gray)
for (sx,sy,sw,sh) in eyes:
	cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
for (sx, sy, sw, sh) in smiles:
         cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
cv2.imshow('Resultant Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
