import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

font = cv2.FONT_HERSHEY_TRIPLEX

class Aula4():
    def escolher(self):

        Aula4.TarefaCasa(object)
        #Aula4.FrameDados(object)

        #Aula4.TarefaTeste(object)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def TarefaCasa(self):
        print("inciando")
        cap = cv2.VideoCapture('dados/aula4/Lancamento_de_dois_dados.mp4')
        success,image = cap.read()
        count = 0
        success = True
        while success:
            success, frame = cap.read()

            #frame = cv2.resize(frame, (width/4, height/4),interpolation=cv2.INTER_AREA)

            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            '''
            print("255: "+str(frame.ravel()[255]))
            print("250: "+str(frame.ravel()[250]))
            print("245: "+str(frame.ravel()[245]))
            print("240: "+str(frame.ravel()[240]))
            print("235: "+str(frame.ravel()[235]))
            '''

            frame = Aula4.EncontraDados(object, frame)
            cv2.imshow('frame', frame)
            '''
            if frame.ravel()[255] >60 and \
                    frame.ravel()[254] > 60 and \
                    frame.ravel()[253] > 60 and \
                    frame.ravel()[252] > 60 and \
                    frame.ravel()[251] > 60 and \
                    frame.ravel()[250] > 60 :
            '''
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def EncontraDados(self, imgInicial):

        img = cv2.cvtColor(imgInicial.copy(), cv2.COLOR_BGR2GRAY)

        kernel = np.ones((13, 13), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        _,thresh = cv2.threshold(img, 245, 250, cv2.THRESH_BINARY_INV)

        #_,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # computes the bounding box for the contour, and draws it on the frame,
        for contour in contours:

            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            if len(approx) >= 6 and len(approx) <= 12 :

                (x, y, w, h) = cv2.boundingRect(contour)

                dados = img[y: y + h, x: x + w]

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                #cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

                img_crop, img_rot = Aula4.crop_rect(object, imgInicial, rect)

                face = Aula4.EncontrarFaces(object, img_crop, dados)

                cv2.putText(imgInicial, face, (x + int(w / 2), y + int(h / 2)), font, 1, (0))

        return imgInicial

    def crop_rect(self, img, rect):
        # get the parameter of the small rectangle
        center, size, angle = rect[0], rect[1], rect[2]
        center, size = tuple(map(int, center)), tuple(map(int, size))

        # get row and col num in img
        height, width = img.shape[0], img.shape[1]

        # calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # rotate the original image
        img_rot = cv2.warpAffine(img, M, (width, height))

        # now rotated rectangle becomes vertical and we crop it
        img_crop = cv2.getRectSubPix(img_rot, size, center)

        return img_crop, img_rot


    def EncontrarFaces(self, img2, subImg):
        height, width = img2.shape[0], img2.shape[1]
        offset = 3
        maxOffset = 200
        minOffset = 30
        if height > offset+4 and width >offset+4 \
            and height < maxOffset and width < maxOffset \
            and height > minOffset and width > minOffset:
            img = img2[offset:width-offset,offset:height-offset].copy()

           # if height > width * 1.8 or width > height * 1.8 :


            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _,thresh = cv2.threshold(gray, 160, 250, cv2.THRESH_BINARY_INV)

            kernel = np.ones((4, 4), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            thresh = cv2.erode(thresh, kernel, iterations=1)

            cv2.imshow("tresh",thresh)
            cv2.imshow("gray",gray)

            #_,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            quantidadeCirculos=0

            for contour in contours:

                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                cv2.drawContours(gray, [approx], 0, (0), 5)

                if len(approx)>=5:
                    quantidadeCirculos += 1
            return str(quantidadeCirculos)

    def FrameDados(self):

        print("inciando")
        img = cv2.imread('dados/aula4/FrameDados2.png',0)
        plt.hist(img.ravel(),256,[0,256])
        plt.show()

        img2 = cv2.imread('dados/aula4/FrameDados3.png',0)
        plt.hist(img2.ravel(),256,[0,256])
        plt.show()
        print("image1")
        print("255: " + str(img.ravel()[255]))
        print("254: " + str(img.ravel()[254]))
        print("253: " + str(img.ravel()[253]))
        print("252: " + str(img.ravel()[252]))
        print("251: " + str(img.ravel()[251]))
        print("250: " + str(img.ravel()[250]))

        print("image2")
        print("255: " + str(img2.ravel()[255]))
        print("254: " + str(img2.ravel()[254]))
        print("253: " + str(img2.ravel()[253]))
        print("252: " + str(img2.ravel()[252]))
        print("251: " + str(img2.ravel()[251]))
        print("250: " + str(img2.ravel()[250]))

    def TarefaTeste(self):
        print("inciando")

        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--file',default='/dados/aula4/Lancamento_de_dois_dados.mp4', help='Path to video file (if not using camera)')
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
                #frame = cv2.resize(frame, (resizeWidth/4, resizeHeight/4),
                #                   interpolation=cv2.INTER_AREA)

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
