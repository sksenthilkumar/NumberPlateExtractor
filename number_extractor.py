import cv2
import numpy as np

debug = True


class NumberboardFinder:
    def __ini__(self, image_folder):
        self.orignalimg = cv2.imread(image_folder)
        print('Hi3')
        if debug:
            cv2.imshow("Original Image", self.orignalimg)

        self.process_image = self.orignalimg

        self.img_gray = cv2.cvtColor(self.process_image, cv2.COLOR_RGB2GRAY)
        if debug:
            cv2.imshow("Gray Converted Image", self.img_gray)
        cv2.waitKey()
        self.process_image = self.img_gray

        self.screenCnt = None
        self.pilot()

    def pilot(self):
        self.noise_removal()
        self.histogram()
        self.threshold()
        self.canny()
        self.dialate()
        self.contour_finding()
        self.final()

    def noise_removal(self):
        # Noise removal with iterative bilateral filter(removes noise while preserving edges)
        noise_removal = cv2.bilateralFilter(self.process_image, 9, 75, 75)
        if debug:
            cv2.imshow("Noise Removed Image", noise_removal)
        cv2.waitKey()
        self.process_image = noise_removal

    def histogram(self):
        # Histogram equalisation for better results
        equal_histogram = cv2.equalizeHist(self.process_image)
        if debug:
            cv2.imshow("After Histogram equalisation", equal_histogram)

        # Morphological opening with a rectangular structure element
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=15)

        cv2.imshow("Morphological opening", morph_image)

        # Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
        sub_morp_image = cv2.subtract(equal_histogram, morph_image)

        cv2.imshow("Subtraction image", sub_morp_image)
        cv2.waitKey()
        self.process_image = sub_morp_image

    def threshold(self):
        # Thresholding the image
        ret, thresh_image = cv2.threshold(self.process_image, 0, 255, cv2.THRESH_OTSU)
        cv2.imshow("Image after Thresholding", thresh_image)
        self.process_image = thresh_image

    def canny(self):
        # Applying Canny Edge detection
        canny_image = cv2.Canny(self.process_image, 250, 255)
        cv2.imshow("Image after applying Canny", canny_image)
        cv2.waitKey()
        self.process_image = canny_image

    def dialate(self):
        canny_image = cv2.convertScaleAbs(self.process_image)
        # dilation to strengthen the edges
        kernel = np.ones((3, 3), np.uint8)
        # Creating the kernel for dilation
        dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
        cv2.imshow("Dilation", dilated_image)
        cv2.waitKey()
        self.process_image = dilated_image

    def contour_finding(self):
        # Finding Contours in the image based on edges
        new, contours, hierarchy = cv2.findContours(self.process_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        # Sort the contours based on area ,so that the number plate will be in top 10 contours

        # loop over our contours
        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:  # Select the contour with 4 corners
                self.screenCnt = approx
                break
        # Drawing the selected contour on the original image

    def final(self):
        final = cv2.drawContours(self.orignalimg, [self.screenCnt], -1, (0, 255, 0), 3)
        cv2.imshow("Image with Selected Contour", final)

        # Masking the part other than the number plate
        mask = np.zeros(self.img_gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [self.screenCnt], 0,  255, -1,)
        cv2.imshow("contour_image", new_image)
        new_image = cv2.bitwise_and(self.orignalimg, self.orignalimg, mask=mask)

        cv2.imshow("Final_image", new_image)

        # Histogram equal for enhancing the number plate for further processing
        y, cr, cb = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2YCrCb))
        # Converting the image to YCrCb model and splitting the 3 channels
        y = cv2.equalizeHist(y)
        # Applying histogram equalisation
        final_image = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)
        # Merging the 3 channels

        cv2.imshow("Enhanced Number Plate", final_image)

        cv2.waitKey()  # Wait for a keystroke from the user


if __name__ == '__main__':
    a = NumberboardFinder()
    a.__ini__('images/img1.png')
