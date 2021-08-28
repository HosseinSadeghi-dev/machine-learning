import cv2

video = cv2.VideoCapture(0)
detector = cv2.QRCodeDetector()

while True:

    validation, frame = video.read()
    if not validation:
        break

    data, bbox, straight_qrcode = detector.detectAndDecode(frame)

    if bbox is not None:
        if data:
            cv2.putText(frame, "QR Code: {}".format(data), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('output', frame)
    if cv2.waitKey(1) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
