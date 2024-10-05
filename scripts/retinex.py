import numpy as np
import cv2


class SingleScaleRetinex:

    def __init__(self):
        self._variance = None

    @property
    def variance(self):
        return self._variance

    @variance.setter
    def variance(self, value):
        self._variance = value

    def ss_retinex(self, img):
        if self._variance:
            img = img+1e-7
            retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), self._variance))
            return retinex
        else:
            print("SSR Error on ss_retinex method: the variance must be set.")
            return None

    def do(self, img):
        if self._variance:
            img = np.float64(img) + 1.0
            img_retinex = self.ss_retinex(img)

            for i in range(img_retinex.shape[2]):
                # Use histogram to get pixel value counts
                hist, bins = np.histogram(img_retinex[:, :, i], bins=1000)  # Adjust bins as needed
                zero_count = hist[np.argmin(np.abs(bins))]

                # Find low_val and high_val using cumulative sum
                cumsum = np.cumsum(hist)
                low_val = bins[np.searchsorted(cumsum, zero_count * 0.1)]
                high_val = bins[np.searchsorted(cumsum, cumsum[-1] - zero_count * 0.1)]

                # Clip values
                img_retinex[:, :, i] = np.clip(img_retinex[:, :, i], low_val, high_val)

                # Normalize
                img_min = np.min(img_retinex[:, :, i])
                img_max = np.max(img_retinex[:, :, i])
                img_retinex[:, :, i] = (img_retinex[:, :, i] - img_min) / (img_max - img_min) * 255

            img_retinex = np.uint8(img_retinex)
            return img_retinex
        else:
            print("SSR Error on do method: the variance must be set.")
            return None


class MultiScaleRetinex:

    def __init__(self, ):
        self._variance = None
        self.SSR = SingleScaleRetinex()

    @property
    def variance(self):
        return self._variance

    @variance.setter
    def variance(self, values: list):
        self._variance = values

    def ms_retinex(self, img):
        if self._variance:
            retinex = np.zeros_like(img)
            for variance in self._variance:
                self.SSR.variance = variance
                retinex += self.SSR.ss_retinex(img)
            retinex = retinex / len(self._variance)
            return retinex
        else:
            print("MSR Error on ms_retinex method: the variance list must be set.")
            return None

    def do(self, img):
        if self._variance:
            img = np.float64(img) + 1.0
            img_retinex = self.ms_retinex(img)

            for i in range(img_retinex.shape[2]):
                # Use histogram for efficiency
                hist, bins = np.histogram(img_retinex[:, :, i], bins=1000)  # Adjust bins as needed
                zero_count = hist[np.argmin(np.abs(bins))]

                # Find low_val and high_val using cumulative sum
                cumsum = np.cumsum(hist)
                low_val = bins[np.searchsorted(cumsum, zero_count * 0.1)]
                high_val = bins[np.searchsorted(cumsum, cumsum[-1] - zero_count * 0.1)]

                # Clip values
                img_retinex[:, :, i] = np.clip(img_retinex[:, :, i], low_val, high_val)

                # Normalize
                img_min = np.min(img_retinex[:, :, i])
                img_max = np.max(img_retinex[:, :, i])
                img_retinex[:, :, i] = (img_retinex[:, :, i] - img_min) / (img_max - img_min) * 255

            img_retinex = np.uint8(img_retinex)
            return img_retinex
        else:
            print("MSR Error on do method: the variance list must be set.")
            return None


if __name__ == "__main__":

    variance_list = [15, 80, 30]
    variance = 300

    img = cv2.imread('sd.jpg')
    SSR = SingleScaleRetinex()
    SSR.variance = variance

    MSR = MultiScaleRetinex()
    MSR.variances = variance_list

    img_ssr = SSR.do(img)
    img_msr = MSR.do(img)

    cv2.imshow('Original', img)
    cv2.imshow('MSR', img_msr)
    cv2.imshow('SSR', img_ssr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()