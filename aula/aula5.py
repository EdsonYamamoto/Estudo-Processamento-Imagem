import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import walk

font = cv2.FONT_HERSHEY_TRIPLEX

class Aula5():
    def escolher(self):
        imgs = Aula5.carregarImagens(object)

        #Aula5.gradienteSudoku(object,"teste", imgs[1])

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
            #print(filenames)
            break

        #print(imgVetor)
        return imgVetor

    def gradienteSudoku(self, nome, imgInicial):
        print("Incializando")
        img = cv2.cvtColor(imgInicial.copy(), cv2.COLOR_BGR2GRAY)
        #cv2.imshow("sudoku["+nome+"]",img)
        '''
        gray = [np.float64(i) for i in img]
        noise = np.random.randn(*img[1].shape)*10
        noisy = [i+noise for i in img]
        noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]
        dst = cv2.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)

        cv2.imshow("denoised ["+nome+"]",dst)
        '''

        tresholdMin = 100
        tresholdMax = 200
        offsetMin = 20
        offsetMax = 150

        kernel = np.ones((8, 8), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        #img = cv2.dilate(img, kernel, iterations=1)

        kernelSize = (5, 5)

        #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernelSize)
        #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelSize)


        #filtro = cv2.GaussianBlur(img,kernelSize,cv2.BORDER_DEFAULT)
        filtro = cv2.GaussianBlur(img,kernelSize,0)

        #cv2.imshow('GaussianBlur['+nome+"]", filtro)

        edges = cv2.Canny(filtro,tresholdMin,tresholdMax)

        #cv2.imshow('Canny['+nome+"]", edges)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i=0
        for contour in contours:
            i+=1
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            if len(approx) >= 4 and len(approx) <= 10:

                (x, y, w, h) = cv2.boundingRect(contour)

                dados = img[y: y + h, x: x + w]

                height, width = dados.shape[0], dados.shape[1]
                if height > offsetMin and width >offsetMin\
                    and height < offsetMax and width < offsetMax:
                    #cv2.imshow("image["+nome+"] Contour["+str(i)+"]", dados)

                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(imgInicial, [box], 0, (0, 0, 255), 1)
        print("imagem pos tratametno")
        cv2.imshow("pos proc["+nome+"]",imgInicial)
        #_,thresh = cv2.threshold(dst, 245, 250, cv2.THRESH_BINARY_INV)

        #cv2.imshow("teste",thresh)
        #cv2.imshow("imgtrans1", imgXGH)