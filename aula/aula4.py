import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

font = cv2.FONT_HERSHEY_COMPLEX

class Aula4():
    def escolher(self):

        Aula4.TarefaCasa(object)
        #Aula4.TarefaTeste(object)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def TarefaCasa(self):
        print("inciando")
        cap = cv2.VideoCapture('dados/aula4/Lancamento_de_dois_dados.mp4')
        while (cap.isOpened()):
            ret, frame = cap.read()

            (height, width) = frame.shape[:2]
            #frame = cv2.resize(frame, (width/4, height/4),interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            print("255: "+str(frame.ravel()[255]))
            print("250: "+str(frame.ravel()[250]))
            print("245: "+str(frame.ravel()[245]))
            print("240: "+str(frame.ravel()[240]))
            print("235: "+str(frame.ravel()[235]))
            if frame.ravel()[255] >50 and \
                    frame.ravel()[250] > 50 and \
                    frame.ravel()[245] > 60 and \
                    frame.ravel()[240] > 60 :
                cv2.imshow('frame', gray)
            #teste = plt.hist(frame.ravel(), 256, [0, 256]);
            #plt.show()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def TarefaTeste(self):
        print("inciando")

        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--file',default='dados/aula4/Lancamento_de_dois_dados.mp4', help='Path to video file (if not using camera)')
        parser.add_argument('-c', '--color', type=str, default='gray',
                            help='Color space: "gray" (default), "rgb", or "lab"')
        parser.add_argument('-b', '--bins', type=int, default=16,
                            help='Number of bins per channel (default 16)')
        parser.add_argument('-w', '--width', type=int, default=0,
                            help='Resize video to specified width in pixels (maintains aspect)')
        args = vars(parser.parse_args())

        # Configure VideoCapture class instance for using camera or file input.
        capture = cv2.VideoCapture("")
        if not args.get('file', False):
            capture = cv2.VideoCapture(0)
        else:
            capture = cv2.VideoCapture(args['file'])

        color = args['color']
        bins = args['bins']
        resizeWidth = args['width']

        # Initialize plot.
        fig, ax = plt.subplots()
        if color == 'rgb':
            ax.set_title('Histogram (RGB)')
        elif color == 'lab':
            ax.set_title('Histogram (L*a*b*)')
        else:
            ax.set_title('Histogram (grayscale)')
        ax.set_xlabel('Bin')
        ax.set_ylabel('Frequency')

        # Initialize plot line object(s). Turn on interactive plotting and show plot.
        lw = 3
        alpha = 0.5
        if color == 'rgb':
            lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha, label='Red')
            lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha, label='Green')
            lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='Blue')
        elif color == 'lab':
            lineL, = ax.plot(np.arange(bins), np.zeros((bins,)), c='k', lw=lw, alpha=alpha, label='L*')
            lineA, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='a*')
            lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='y', lw=lw, alpha=alpha, label='b*')
        else:
            lineGray, = ax.plot(np.arange(bins), np.zeros((bins, 1)), c='k', lw=lw, label='intensity')
        ax.set_xlim(0, bins - 1)
        ax.set_ylim(0, 1)
        ax.legend()
        plt.ion()
        plt.show()

        # Grab, process, and display video frames. Update plot line object(s).
        while True:
            (grabbed, frame) = capture.read()

            if not grabbed:
                break

            # Resize frame to width, if specified.
            if resizeWidth > 0:
                (height, width) = frame.shape[:2]
                resizeHeight = int(float(resizeWidth / width) * height)
                frame = cv2.resize(frame, (resizeWidth/4, resizeHeight/4),
                                   interpolation=cv2.INTER_AREA)

            # Normalize histograms based on number of pixels per frame.
            numPixels = np.prod(frame.shape[:2])
            if color == 'rgb':
                cv2.imshow('RGB', frame)
                (b, g, r) = cv2.split(frame)
                histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
                histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
                histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
                lineR.set_ydata(histogramR)
                lineG.set_ydata(histogramG)
                lineB.set_ydata(histogramB)
            elif color == 'lab':
                cv2.imshow('L*a*b*', frame)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                (l, a, b) = cv2.split(lab)
                histogramL = cv2.calcHist([l], [0], None, [bins], [0, 255]) / numPixels
                histogramA = cv2.calcHist([a], [0], None, [bins], [0, 255]) / numPixels
                histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
                lineL.set_ydata(histogramL)
                lineA.set_ydata(histogramA)
                lineB.set_ydata(histogramB)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Grayscale', gray)
                histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
                lineGray.set_ydata(histogram)
            fig.canvas.draw()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    def EncontrarFaces(self, nome , img, subImg):

        _, thresh = cv2.threshold(subImg, 127, 255, cv2.THRESH_BINARY_INV)
        print("nome:",nome)
        if nome is '1' :
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