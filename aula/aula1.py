import cv2
import json
import time

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

        print(imgXGH.shape)
        cv2.imshow("imgtrans1", imgXGH)

        print("*****************************************************")

        vetorImagens=[]
        for x in range(0, len(config)):

            img = cv2.imread(config[x]['caminho'])

            imgCopy = img[0:400, 0:400].copy()
            imgCopy[0:200, 0:200] = imgXGH

            print("Shape: ", img.shape)
            print("Size: ", img.size)
            print("Type: ", img.dtype)


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


        cv2.waitKey()
        cv2.destroyAllWindows()