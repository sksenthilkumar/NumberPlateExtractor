import glob
import cv2
import pytesseract
import numpy as np
labels = ["SDN7484U", "GT5861K", "PA9588S", "SJG5465D", "SHA3085K", "YN1219K", "SBS6216G",
"SEP1", "QX30K", "SPF1", "SBA1234A", "SV2872X", "YB1234A", "SBA1234A", "SBA1234A"]
image_path = "/home/senthil/projects/NumberPlateExtractor/images/singapore_numberPlates/"
Show_image_processing = False

files = glob.glob(image_path + "*")
files = np.array(sorted(files))
image_name = [x.replace(image_path, "") for x in files]
results = []
acc_result = np.zeros((len(files), len(range(1, 14))))
for i, imPath in enumerate(files):
    text_in_image = []

    print(image_name[i])
    im = cv2.imread(imPath, cv2.IMREAD_GRAYSCALE)
    im = cv2.bilateralFilter(im, 9, 75, 75)
    ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
    if Show_image_processing:
        cv2.imshow("im", im)
        cv2.waitKey()
        cv2.destroyAllWindows()

    for j in range(1, 14):
        config = ('-l eng --oem 3 --psm %i' % j)
        # Read image from disk


        # Run tesseract OCR on image
        try:
            text = pytesseract.image_to_string(im, config=config)
            text = text.replace(" ", "")
        except:
            text = "ERROR"

        text_in_image.append(text if text != '' else 'NA')
        if labels[i] in text:
            acc_result[i][j-1] = 1

        else:
            acc_result[i][j-1] = 0

    print("True label: %s" % labels[i])
    print("Predictions of the image %s:\n %s" % (image_name[i], text_in_image))
    # print("approx_match : %i \nexact_match : %i \n" %(approx_match, runtime_error))

    results.append(text_in_image)

print("The accuracy matrix for psm value and images \n", acc_result)

print("Number of correct predictions for each image:")
for i in range(acc_result.shape[0]):
    print(image_name[i], ":", np.sum(acc_result[i,:]))

print("Number of correct predictions for each psm value:")
for i in range(acc_result.shape[1]):
    print("psm = %i : %i" % (i, np.sum(acc_result[:, i])))
