import cv2
import numpy as np
from os import walk

font = cv2.FONT_HERSHEY_TRIPLEX

class Aula5():
    def escolher(self):
        imgs = Aula5.carregarImagens(object)

        for i in range(0,len(imgs)):
            Aula5.gradienteSudoku(object,str(i),imgs[i])

        cv2.waitKey()
        cv2.destroyAllWindows()


    def carregarImagens(self):
        caminho = "dados/aula5/sudoku/"

        f = []
        imgVetor = []
        for (dirpath, dirnames, filenames) in walk(caminho):
            for nome in range(len(filenames)):

                img = cv2.imread(caminho+filenames[nome])
                imgVetor.append(img)
            f.extend(filenames)
            break

        return imgVetor

    def gradienteSudoku(self, nome, imgInicial):
        print("Incializando")
        img = cv2.cvtColor(imgInicial.copy(), cv2.COLOR_BGR2GRAY)

        offsetMin = 17
        offsetMax = 150

        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)

        kernelSize = (1, 1)

        filtro = cv2.GaussianBlur(img,kernelSize,0)

        _, thresh = cv2.threshold(filtro, 200, 250, cv2.THRESH_BINARY)

        cv2.imshow("pos trash["+nome+"]",thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i=0
        for contour in contours:
            i+=1
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            if len(approx) >= 4 and len(approx) <= 7:

                (x, y, w, h) = cv2.boundingRect(contour)

                dados = img[y: y + h, x: x + w]

                height, width = dados.shape[0], dados.shape[1]
                if height > offsetMin and width >offsetMin\
                    and height < offsetMax and width < offsetMax:

                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(imgInicial, [box], 0, (0, 0, 255), 1)

                    temNumero = Aula5.encontraNumero(object, dados)
                    if temNumero is False:
                        cv2.drawContours(imgInicial, [box], 0, (0, 0, 255), 1)
                    else:
                        cv2.drawContours(imgInicial, [box], 0, (255, 0, 0), 3)

        cv2.imshow("pos proc["+nome+"]",imgInicial)
    def encontraNumero(self, img):
        _, thresh = cv2.threshold(img, 200, 250, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            if len(approx) >= 5 :
                return True
        return False