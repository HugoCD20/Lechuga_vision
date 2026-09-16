import cv2

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

print("✅ Cámara abierta correctamente")

while True:
    ret, frame = camera.read()

    if not ret:
        print("❌ No se pudo leer el frame")
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()