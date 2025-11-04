import cv2

# load cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# load image
image = cv2.imread("your_image_path_here.jpg")

if image is None:
    print("Image not found")
    exit()

# convert image to gray (haar needs gray)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# draw boxes
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Number of faces detected:", len(faces))
