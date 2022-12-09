import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
image = cv.imread('aldeano.png', cv.IMREAD_UNCHANGED)

faceClassif = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv.resize(frame, (640, 500)) 
    faces = faceClassif.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        nuevo_tamanio = cv.resize(image, (w, h))   
        filas = nuevo_tamanio.shape[0]
        columnas = w
        altura_rostro = filas // 1
        if y + altura_rostro - filas >= 0:
            numero_frame = frame[y + altura_rostro - filas: y + altura_rostro,
                                 x: x + columnas]
        else:
            numero_frame = frame[0: y + altura_rostro,
                                 x: x + columnas]

        mask = nuevo_tamanio[:, :, 3]
        mask_inv = cv.bitwise_not(mask)

        black_image = cv.bitwise_and(nuevo_tamanio, nuevo_tamanio, mask=mask)
        black_image = black_image[0:, :, 0:3]
        nuevo_frame = cv.bitwise_and(
            numero_frame, numero_frame, mask=mask_inv[0:, :])

        imagen_final = cv.add(black_image, nuevo_frame)
        if y + altura_rostro - filas >= 0:
            frame[y + altura_rostro - filas: y +
                  altura_rostro, x: x + columnas] = imagen_final

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv.destroyAllWindows()
