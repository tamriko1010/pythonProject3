import cv2

#cap = cv2.VideoCapture('photo.mp4')
cap = cv2.VideoCapture(0)
find_face = cv2.CascadeClassifier('model_face.xml')
find_eyes = cv2.CascadeClassifier('eyes.xml')
while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = find_face.detectMultiScale(gray, 1.1, 2)
    for(x, y, w, h) in faces:
        eyes = find_eyes.detectMultiScale(gray, 4, 2)
        if len(eyes) >= 2:
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            if eyes[0][0] < eyes[1][0]:
                eyes = cv2.rectangle(img, (x, ey1), (x + w, ey1 + eh2), (0,255,0), thickness=2)
                mask_eye = eyes[ey1:ey1 + eh2, x: x + w]
                mask_eye = cv2.GaussianBlur(mask_eye, (69,59), 0)
                img[ey1:ey1 + eh2, x: x + w] = mask_eye

    cv2.imshow('Result', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break