import cv2

if __name__ == '__main__':
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img1 = cv2.imread("dat.jpg")#dat
    img2 = cv2.imread("quynh.jpg")#quynh
    img3 = cv2.imread("long.jpg")#long
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray1, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.imwrite('esp_image/processed_image.jpg', cv2.resize(gray1[y:y + h, x:x + w], (12, 12)))
    faces = facedetect.detectMultiScale(gray1, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.imwrite('esp_image/processed_image.jpg', cv2.resize(gray2[y:y + h, x:x + w], (12, 12)))
    faces = facedetect.detectMultiScale(gray1, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.imwrite('esp_image/processed_image.jpg', cv2.resize(gray3[y:y + h, x:x + w], (12, 12)))