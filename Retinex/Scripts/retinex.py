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
            print("Error on ss_retinex: the variance must be set.")
            return None

    def do_ssr(self, img):
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

    def do_ssr_orig(self, img):
        if self._variance:
            img = np.float64(img) + 1.0
            img_retinex = self.ss_retinex(img)
            for i in range(img_retinex.shape[2]):
                unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
                for u, c in zip(unique, count):
                    if u == 0:
                        zero_count = c
                        break
                low_val = unique[0] / 100.0
                high_val = unique[-1] / 100.0
                for u, c in zip(unique, count):
                    if u < 0 and c < zero_count * 0.1:
                        low_val = u / 100.0
                    if u > 0 and c < zero_count * 0.1:
                        high_val = u / 100.0
                        break
                img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

                img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                                       (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                                       * 255

            img_retinex = np.uint8(img_retinex)
            return img_retinex
        else:
            print("Error on do_ssr: the variance must be set.")
            return None


class MultiScaleRetinex:

    def __init__(self, ):
        self._variances = None
        self.SSR = SingleScaleRetinex()

    @property
    def variances(self):
        return self._variances

    @variances.setter
    def variances(self, values: list):
        self._variances = values

    def ms_retinex(self, img):
        if self._variances:
            retinex = np.zeros_like(img)
            for variance in self._variances:
                self.SSR.variance = variance
                retinex += self.SSR.ss_retinex(img)
            retinex = retinex / len(self._variances)
            return retinex
        else:
            print("Error on ms_retinex: the variance list must be set.")
            return None

    def do_msr(self, img):
        if self._variances:
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
            print("Error on do_msr: the variance must be set.")
            return None

    def do_msr_orig(self, img):
        if self._variances:
            img = np.float64(img) + 1.0
            img_retinex = self.ms_retinex(img)
            for i in range(img_retinex.shape[2]):
                unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
                for u, c in zip(unique, count):
                    if u == 0:
                        zero_count = c
                        break
                low_val = unique[0] / 100.0
                high_val = unique[-1] / 100.0
                for u, c in zip(unique, count):
                    if u < 0 and c < zero_count * 0.1:
                        low_val = u / 100.0
                    if u > 0 and c < zero_count * 0.1:
                        high_val = u / 100.0
                        break
                img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

                img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                                       (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                                       * 255
            img_retinex = np.uint8(img_retinex)
            return img_retinex
        else:
            print("Error on do_msr: the variance must be set.")
            return None
