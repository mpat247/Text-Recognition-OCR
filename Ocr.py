import pytesseract
import cv2

def main():
    config = ('--oem 1 --psm 4')

    for i in range(1, 89178):
        imPath = './dataset/word_' + str(i) + '.png'
        image = cv2.imread(imPath, cv2.IMREAD_COLOR)
        prep = preprocessing(image)

        #cv2.imshow("prep", prep)
        #cv2.waitKey(0)

        text = pytesseract.image_to_string(prep, config=config)
        print(text)

        if text == "":
            text = "\n"
        outfilePath = './output/word_' + str(i) + '.txt'
        f = open(outfilePath, "w")
        f.write(text)
        f.close()


def preprocessing(image):
    # kernel = np.ones((2, 2), np.uint8)
    newIm = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    newIm = greyscale(newIm)
    newIm = cv2.GaussianBlur(newIm, (5, 5), 3)
    newIm = cv2.adaptiveThreshold(newIm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # newIm = cv2.Canny(newIm, 100, 200)
    # newIm = cv2.morphologyEx(newIm, cv2.MORPH_OPEN, kernel)
    # newIm = cv2.morphologyEx(newIm, cv2.MORPH_CLOSE, kernel)
    # newIm = cv2.bitwise_not(newIm)
    return newIm


def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
    main()