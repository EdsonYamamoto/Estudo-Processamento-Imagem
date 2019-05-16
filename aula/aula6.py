import cv2
import numpy as np

mog = cv2.createBackgroundSubtractorMOG2()
kernelSize = 5


font = cv2.FONT_HERSHEY_TRIPLEX

class Aula6():
    def escolher(self):
        img = Aula6.carregarFundo(object)
        cap = Aula6.carregarVideo(object)

        Aula6.croma_key_teste3(object,cap,img)

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

    def croma_key_teste3(self, cap, img):
        success = True
        while success:
            success, frame = cap.read()
            cols, rows = frame.shape[:2]
            background = img[0:cols, 0:rows]
            frame = cv2.flip(frame, 1, frame)

            hls_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

            hue = hls_image[:, :, 0]

            binary_hue = cv2.inRange(hue, 50, 100)

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
        cap = cv2.VideoCapture('dados/aula6/Green_Screen_Sample_1.mp4')
        return cap

    def carregarFundo(self):
        img = cv2.imread('dados/aula6/fundo1.jpg')
        return img
