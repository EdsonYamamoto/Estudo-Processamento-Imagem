import cv2
import numpy as np

class Aula2():
    def escolher(self):
        texto = "[0] tarefa para casa\n[1] teste"

        print(texto)
        teste = input()
        if teste is '1':
            Aula2.test1(object)
        if teste is '0':
            Aula2.TarefaCasa(object)

    def test1(self):
        img = cv2.imread('dados/aula1/1.png')
        cv2.imshow('Exemplo', img)
        print("Shape: ", img.shape)
        print("Size: ", img.size)
        print("Type: ", img.dtype)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def TarefaCasa(self):
        inputData = np.array([
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
        ], dtype=np.uint8
        )
        data = np.copy(inputData)


        resultado = 0

        kernel = np.array((
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]), dtype="int")
        print(data)
        print(kernel)
        for x in range(1, data.shape[0]-1):
            for y in range(1, data.shape[1]-1):
                if np.array_equal(kernel, data[x-1:x+2,y-1:y+2]):
                    resultado = resultado + 1
                    for xAux in range(-1, 2):
                        for yAux in range(-1, 2):
                            data[x+xAux][y+yAux] = 2


        for x in range (0,inputData.shape[0]):
            for y in range (0,inputData.shape[1]):
                if data[x][y] == 1:
                    data[x][y] = 255
                if inputData[x][y] == 1:
                    inputData[x][y] = 255

        for x in range (0,data.shape[0]):
            for y in range (0,data.shape[1]):
                if data[x][y] == 2:
                    data[x][y] = 124


        output_image = data
        rate = 50

        kernel = (kernel + 1) * 127
        kernel = np.uint8(kernel)
        kernel = cv2.resize(kernel, None, fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("kernel", kernel)
        cv2.moveWindow("kernel", 0, 0)

        input_image = cv2.resize(inputData, None, fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Original", input_image)
        cv2.moveWindow("Original", 0, 200)

        output_image = cv2.resize(output_image, None, fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Hit or Miss", output_image)
        cv2.moveWindow("Hit or Miss", 500, 200)

        cv2.waitKey()
        cv2.destroyAllWindows()