import glob
import cv2
import pytesseract
import numpy as np
labels = ["SDN7484U", "GT5861K", "PA9588S", "SJG5465D", "SHA3085K", "YN1219K", "SBS6216G",
"SEP1", "QX30K", "SPF1", "SBA1234A", "SV2872X", "YB1234A", "SBA1234A", "SBA1234A"]


image_path = "/home/senthil/projects/NumberPlateExtractor/images/singapore_numberPlates/"
files = glob.glob(image_path + "*")
files = np.array(sorted(files))
image_name = [x.replace(image_path, "") for x in files]
for i, imPath in enumerate(files):
    im = cv2.imread(imPath, cv2.IMREAD_GRAYSCALE)
    print(im.size)
results = []
acc_result = np.zeros((len(files), len(range(1, 14))))
for i, imPath in enumerate(files):
    image = []
    approx_match = 0
    exact_match = 0
    no_results = 0
    runtime_error = 0
    print("image %i" %(i+1))
    im = cv2.imread(imPath, cv2.IMREAD_GRAYSCALE)
    im = cv2.bilateralFilter(im, 9, 75, 75)
    ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
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
        # Print recognized text
        image.append(text if text != '' else 'NA')
        if labels[i] in text:
            approx_match = approx_match + 1
            acc_result[i][j-1] = 1
            if labels[i] == text:
                exact_match = exact_match + 1

        else:
            acc_result[i][j-1] = 0




        # print("%i : %i -> %s" %i+1, im.size, text))
    print(labels[i])
    print(image)
    # print("approx_match : %i \nexact_match : %i \n" %(approx_match, runtime_error))

    results.append(image)

print(acc_result)

for i in range(acc_result.shape[0]):
    print(np.sum(acc_result[i,:]))

pms_performance = []
print('ya')
for i in range(acc_result.shape[1]):
    print(np.sum(acc_result[:, i]))
