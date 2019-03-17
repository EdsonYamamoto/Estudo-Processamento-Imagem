import cv2
import json
import time
import numpy as np

class Aula():
    def escolher(self):
        texto = "[0] tarefa para casa"

        print(texto)
        teste = input()
        if teste is '1':
            Aula.test1(object)
        if teste is '0':
            Aula.TarefaCasa(object)

    def test1(self):
        img = cv2.imread('dados/aula1/1.png')
        cv2.imshow('Exemplo', img)
        print("Shape: ", img.shape)
        print("Size: ", img.size)
        print("Type: ", img.dtype)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def TarefaCasa(self):
        print("*****************************************************")
        print("Carregamento do JSON de configuração")
        with open('dados/aula1/vetorCaminhoImagem.json') as f:
            config = json.load(f)
        print(json.dumps(config, indent=4, sort_keys=True))
        print("Carregamento do JSON de configuração - Finalizado")
        print("*****************************************************")

        print("*****************************************************")
        print("Carregamento do XGH ")
        imgXGHOrig = cv2.imread('dados/aula1/xgh.png',-1)

        imgXGH= imgXGHOrig[0:200, 0:200].copy()

        (wH, wW) = imgXGH.shape[:2]

        (frame_h, frame_w) = imgXGH.shape[:2]

        (B, G, R, A) = cv2.split(imgXGH)
        B = cv2.bitwise_and(B, B, mask=A)
        G = cv2.bitwise_and(G, G, mask=A)
        R = cv2.bitwise_and(R, R, mask=A)
        watermark = cv2.merge([B, G, R, A])

        #cv2.imshow("imgtrans1", imgXGH)

        print("*****************************************************")

        vetorImagens=[]
        for x in range(0, len(config)):

            img = cv2.imread(config[x]['caminho'])

            imgCopy = img[0:400, 0:400].copy()
            #imgCopy[0:200, 0:200] = imgXGH

            print("Shape: ", img.shape)
            print("Size: ", img.size)
            print("Type: ", img.dtype)

            (h, w) = imgCopy.shape[:2]
            imgCopy = np.dstack([imgCopy, np.ones((h, w), dtype="uint8") * 255])

            overlay = np.zeros((h, w, 4), dtype="uint8")
            overlay[h - wH - 10:h - 10, w - wW - 10:w - 10] = watermark

            imgCopy = cv2.addWeighted(overlay, 0.25, imgCopy, 1.0, 0)
            #imgCopy = cv2.addWeighted(imgCopy[x + 1], 1.0, vetorImagens[x], 1 - fadein, 0)

            top = 20
            botton = 20
            left = 20
            right = 20

            color = [101, 52, 152]

            imgCopy = cv2.copyMakeBorder(imgCopy, top, botton, left, right, cv2.BORDER_CONSTANT, value=color)


            vetorImagens.append(imgCopy)
        while True:
            for x in range(0, len(vetorImagens)):
                if x < len(vetorImagens)-1:
                    for IN in range(0, 11):
                        fadein = IN / 10.0
                        img = cv2.addWeighted(vetorImagens[x+1], fadein, vetorImagens[x], 1-fadein, 0)
                        #print(fadein)

                        cv2.imshow("imgtrans", img)
                        cv2.waitKey(1)

                        time.sleep(0.1)
                        if fadein == 1.0:  # blendmode mover
                            fadein = 1.0

                    time.sleep(2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return


        cv2.waitKey()
        cv2.destroyAllWindows()