import cv2
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX

class Aula3():
    def escolher(self):

        Aula3.TarefaCasa(object)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def TarefaCasa(self):
        print("inciando")

        imgInicial = Aula3.CarregarImagem(object)

        img = cv2.cvtColor(imgInicial.copy(), cv2.COLOR_BGR2GRAY)

        kernel = np.ones((7, 7), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)

        # binary image
        _, thresh = cv2.threshold(img, 250, 250, cv2.THRESH_BINARY_INV)
        cv2.imshow("Thresh ", thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = img.shape
        min_x, min_y = width, height
        max_x = max_y = 0
        offset = 5

        # computes the bounding box for the contour, and draws it on the frame,
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            print("*****************************************************")
            print("x = ", x, " y= ", y, " w = ", w, " h = ", h)

            dados = img[y: y + h, x: x + w]

            face = Aula3.EncontrarFaces(object, str(contours.index(contour)), img, dados)

            cv2.putText(imgInicial, face, (x+int(w/2), y+int(h/2)), font, 1, (0))


        cv2.imshow("ImagemFinal ", imgInicial)


    def CarregarImagem(self):
        img = cv2.imread('dados/aula3/dados.jpg')
        print("Shape: ", img.shape)
        print("Size: ", img.size)
        print("Type: ", img.dtype)
        return img

    def EncontrarFaces(self, nome , img, subImg):

        _, thresh = cv2.threshold(subImg, 127, 255, cv2.THRESH_BINARY_INV)
        print("nome:",nome)
        cv2.imshow("Face1: " + nome , subImg)
        cv2.imshow("TrashFace1: " + nome , thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = subImg.shape
        min_x, min_y = width, height
        max_x = max_y = 0

        quantidadeCirculos=0

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)


            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            cv2.drawContours(subImg, [approx], 0, (0), 5)
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            if len(approx)>=5:
                quantidadeCirculos += 1
        return str(quantidadeCirculos)
    def Teste(self):


        im = Aula3.CarregarImagem(object)

        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create()

        # Detect blobs.
        keypoints = detector.detect(im)

        print("Blobs = ", len(keypoints))
        for marker in keypoints:
            # center
            x, y = np.int(marker.pt[0]), np.int(marker.pt[1])
            pos = np.int(marker.size / 2)
            cv2.circle(im, (x, y), 3, 255, -1)
            cv2.rectangle(im, (x - pos, y - pos), (x + pos, y + pos), 0, 1)

        cv2.imshow("Blobs = " + str(len(keypoints)), im)