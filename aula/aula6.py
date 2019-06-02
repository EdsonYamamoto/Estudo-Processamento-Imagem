import cv2
import numpy as np

mog = cv2.createBackgroundSubtractorMOG2()
kernelSize = 5


font = cv2.FONT_HERSHEY_TRIPLEX

class Aula6():
    def escolher(self):
        img = Aula6.carregarFundo(object)
        cap, verde = Aula6.carregarVideo(object)

        Aula6.croma_key_teste3(object,cap,img, verde)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def croma_key_teste1(self,cap,img):
        success = True
        while success:
            success, frame = cap.read()

            img = (255 - img)

            final = frame - img

            cv2.imshow('frame', final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def croma_key_teste2(self, cap, img):
        success = True
        while success:

            success, frame = cap.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = mog.apply(frame)

            gaussianBlur = cv2.GaussianBlur(mask, (11, 11), 3.5)
            ret, thresh = cv2.threshold(gaussianBlur, 10, 255, cv2.THRESH_BINARY)

            cv2.imshow('Camera', frame)
            cv2.imshow('Mask', thresh)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def croma_key_teste3(self, cap, img, verde):
        success = True
        while success:
            success, frame = cap.read()
            cols, rows = frame.shape[:2]
            background = img[0:cols, 0:rows]
            frame = cv2.flip(frame, 1, frame)

            hls_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

            hls_image = cv2.medianBlur(hls_image, 5)

            hue = hls_image[:, :, 0]
            #Green
            lower_green = 50
            upper_green = 60

            lower_blue = 110
            upper_blue = 130
            #blue

            if verde:
               binary_hue = cv2.inRange(hue, lower_green, upper_green)
            else:
               binary_hue = cv2.inRange(hue, lower_blue, upper_blue)

            mask = np.zeros(hls_image.shape, dtype=np.uint8)

            mask[:, :, 0] = binary_hue
            mask[:, :, 1] = binary_hue
            mask[:, :, 2] = binary_hue

            blured = cv2.GaussianBlur(mask, (11, 11), 0)
            blured_inverted = cv2.bitwise_not(blured)


            bg_key = cv2.bitwise_and(background, blured)
            fg_key = cv2.bitwise_and(frame, blured_inverted)

            cv2.imwrite('bg.jpg', bg_key)
            cv2.imwrite('fg.jpg', fg_key)
            keyed = cv2.add(bg_key, fg_key)
            cv2.imshow('frame', keyed)

            k = cv2.waitKey(33)
            if k == 27:  # ESC
                break


    def carregarVideo(self):
        print("[0]Blue\n[1]Green video 1\n[2]Green video 2\n[3]Green video 3\n")
        key = input()
        verde = False

        if key is "0":
            cap = cv2.VideoCapture('dados/aula6/Blue_Screen_Sample.mp4')

        if key is "1":
            cap = cv2.VideoCapture('dados/aula6/Green_Screen_Sample_1.mp4')
            verde = True


        if key is "2":
            cap = cv2.VideoCapture('dados/aula6/Green_Screen_Sample_2.mp4')
            verde = True

        if key is "3":
            cap = cv2.VideoCapture('dados/aula6/Green_Screen_Sample_3.mp4')
            verde = True


        return cap, verde

    def carregarFundo(self):
        pasta="dados/aula6/"
        print("Escolha um fundo")
        key = input()
        if key is "1":
            img = cv2.imread(pasta+'fundo1.jpg')
        if key is "2":
            img = cv2.imread(pasta+'fundo2.jpg')
        if key is "3":
            img = cv2.imread(pasta+'fundo3.jpg')
        return img
