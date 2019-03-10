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

        vetorImagens=[]
        for x in range(0, len(config)):

            img = cv2.imread(config[x]['caminho'])

            print("Shape: ", img.shape)
            print("Size: ", img.size)
            print("Type: ", img.dtype)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            vetorImagens.append(img[0:200,200:400].copy())
        for x in range(0, len(vetorImagens)):
            if x < len(vetorImagens)-1:
                for IN in range(0, 10):
                    fadein = IN / 10.0
                    dst = cv2.addWeighted(vetorImagens[x], 0.5, vetorImagens[x+1], 0.3, 0)
                    cv2.imshow('imgFim', dst)
                    cv2.waitKey(1)

                    time.sleep(0.05)
                    if fadein == 1.0:  # blendmode mover
                        fadein = 1.0

                time.sleep(2)

        cv2.waitKey()
        cv2.destroyAllWindows()