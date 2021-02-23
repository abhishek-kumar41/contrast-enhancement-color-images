from pathlib import Path
import numpy
import skimage.io
import skimage.color
from matplotlib import pyplot
import cv2
import pywt
from skimage.metrics import structural_similarity


def contrast_enhancement():

    image_path = Path('img21.jpg')
    # image_path = Path('img7.jpg')
    # image_path = Path('img12.jpg')

    image = skimage.io.imread(image_path.as_posix())
    image = image/255.0

    hsv_image = rgb2hsv(image)
    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    v = hsv_image[:, :, 2]

    v_dwt = pywt.dwt2(v, 'haar')
    cA, (cH, cV, cD) = v_dwt

    h_cA, w_cA = cA.shape
    v_dwt_matrix = numpy.zeros((2*h_cA, 2*w_cA), dtype=float)
    h_vwt, w_vwt = v_dwt_matrix.shape

    v_dwt_matrix[:h_cA, :w_cA] = numpy.copy(cA)
    v_dwt_matrix[:h_cA, w_cA:w_vwt] = numpy.copy(cH)
    v_dwt_matrix[h_cA: h_vwt, :w_cA] = numpy.copy(cV)
    v_dwt_matrix[h_cA: h_vwt, w_cA:w_vwt] = numpy.copy(cD)
    min_cA = numpy.min(cA)
    max_cA = numpy.max(cA)

    cA2 = cA/max_cA
    new_cA = clahe_(cA2)/255.0
    new_cA = new_cA*(max_cA - min_cA)

    v_dwt_matrix_new = numpy.zeros(v_dwt_matrix.shape, dtype=float)
    v_dwt_matrix_new[:h_cA, :w_cA] = numpy.copy(new_cA)
    v_dwt_matrix_new[:h_cA, w_cA:w_vwt] = numpy.copy(cH)
    v_dwt_matrix_new[h_cA: h_vwt, :w_cA] = numpy.copy(cV)
    v_dwt_matrix_new[h_cA: h_vwt, w_cA:w_vwt] = numpy.copy(cD)

    v_idwt = pywt.idwt2((new_cA, (cH, cV, cD)), 'haar')

    s_enhanced = clahe_(s)/255.0

    height, width = h.shape
    enhanced_image = numpy.zeros((height, width, 3), dtype=float)
    enhanced_image[:, :, 0] = h
    enhanced_image[:, :, 1] = s_enhanced
    enhanced_image[:height, :width, 2] = numpy.copy(v_idwt[:height, :width])

    rgb_image_idwt_hsv = hsv2rgb(enhanced_image)

    clahe_rgb = clahe_on_rgb(image)

    ycbcr_image = rgb2ycbcr(image)
    y = ycbcr_image[:, :, 0]
    y = clahe_(y)/255
    ycbcr_image[:, :, 0] = y
    # cb = ycbcr_image[:, :, 1]
    # cb = clahe_(cb) / 255
    # ycbcr_image[:, :, 1] = cb
    # cr = ycbcr_image[:, :, 2]
    # cr = clahe_(cr) / 255
    # ycbcr_image[:, :, 2] = cr
    rgb_from_ycbcr = ycbcr2rgb(ycbcr_image)

    [mse_rgb_r, psnr_rgb_r] = mse_(image[:, :, 0], clahe_rgb[:, :, 0])
    [mse_ycbcr_r, psnr_ycbcr_r] = mse_(image[:, :, 0], rgb_from_ycbcr[:, :, 0])
    [mse_hsv_r, psnr_hsv_r] = mse_(image[:, :, 0], rgb_image_idwt_hsv[:, :, 0])

    [mse_rgb_g, psnr_rgb_g] = mse_(image[:, :, 1], clahe_rgb[:, :, 1])
    [mse_ycbcr_g, psnr_ycbcr_g] = mse_(image[:, :, 1], rgb_from_ycbcr[:, :, 1])
    [mse_hsv_g, psnr_hsv_g] = mse_(image[:, :, 1], rgb_image_idwt_hsv[:, :, 1])

    [mse_rgb_b, psnr_rgb_b] = mse_(image[:, :, 2], clahe_rgb[:, :, 2])
    [mse_ycbcr_b, psnr_ycbcr_b] = mse_(image[:, :, 2], rgb_from_ycbcr[:, :, 2])
    [mse_hsv_b, psnr_hsv_b] = mse_(image[:, :, 2], rgb_image_idwt_hsv[:, :, 2])

    ssim_rgb_r = structural_similarity(image[:, :, 0], clahe_rgb[:, :, 0])
    ssim_rgb_g = structural_similarity(image[:, :, 1], clahe_rgb[:, :, 1])
    ssim_rgb_b = structural_similarity(image[:, :, 2], clahe_rgb[:, :, 2])

    ssim_ycbcr_r = structural_similarity(image[:, :, 0], rgb_from_ycbcr[:, :, 0])
    ssim_ycbcr_g = structural_similarity(image[:, :, 1], rgb_from_ycbcr[:, :, 1])
    ssim_ycbcr_b = structural_similarity(image[:, :, 2], rgb_from_ycbcr[:, :, 2])

    ssim_hsv_r = structural_similarity(image[:, :, 0], rgb_image_idwt_hsv[:, :, 0])
    ssim_hsv_g = structural_similarity(image[:, :, 1], rgb_image_idwt_hsv[:, :, 1])
    ssim_hsv_b = structural_similarity(image[:, :, 2], rgb_image_idwt_hsv[:, :, 2])

    # ssim_rgb_r = ssim_(image[:, :, 0], clahe_rgb[:, :, 0])
    # ssim_rgb_g = ssim_(image[:, :, 1], clahe_rgb[:, :, 1])
    # ssim_rgb_b = ssim_(image[:, :, 2], clahe_rgb[:, :, 2])
    #
    # ssim_ycbcr_r = ssim_(image[:, :, 0], rgb_from_ycbcr[:, :, 0])
    # ssim_ycbcr_g = ssim_(image[:, :, 1], rgb_from_ycbcr[:, :, 1])
    # ssim_ycbcr_b = ssim_(image[:, :, 2], rgb_from_ycbcr[:, :, 2])
    #
    # ssim_hsv_r = ssim_(image[:, :, 0], rgb_image_idwt_hsv[:, :, 0])
    # ssim_hsv_g = ssim_(image[:, :, 1], rgb_image_idwt_hsv[:, :, 1])
    # ssim_hsv_b = ssim_(image[:, :, 2], rgb_image_idwt_hsv[:, :, 2])
    print('Between Original Image and Contrast Enhanced Image by CLAHE:')
    print(' ')
    print(f'MSE_R = {mse_rgb_r}, MSE_G = {mse_rgb_g}, MSE_B = {mse_rgb_b}')
    print(f'PSNR_R = {psnr_rgb_r}, PSNR_G = {psnr_rgb_g}, PSNR_B = {psnr_rgb_b}')
    print(f'SSIM_R = {ssim_rgb_r}, SSIM_G = {ssim_rgb_g}, SSIM_B = {ssim_rgb_b}')
    print(' ')
    print(' ')
    print('Between Original Image and Contrast Enhanced Image using YCbCr colour map')
    print(' ')
    print(f'MSE_R = {mse_ycbcr_r}, MSE_G = {mse_ycbcr_g}, MSE_B = {mse_ycbcr_b}')
    print(f'PSNR_R = {psnr_ycbcr_r}, PSNR_G = {psnr_ycbcr_g}, PSNR_B = {psnr_ycbcr_b}')
    print(f'SSIM_R = {ssim_ycbcr_r}, SSIM_G = {ssim_ycbcr_g}, SSIM_B = {ssim_ycbcr_b}')
    print(' ')
    print(' ')
    print('Between Original Image and Contrast Enhanced Image using HSV colour map')
    print(' ')
    print(f'MSE_R = {mse_hsv_r}, MSE_G = {mse_hsv_g}, MSE_B = {mse_hsv_b}')
    print(f'PSNR_R = {psnr_hsv_r}, PSNR_G = {psnr_hsv_g}, PSNR_B = {psnr_hsv_b}')
    print(f'SSIM_R = {ssim_hsv_r}, SSIM_G = {ssim_hsv_g}, SSIM_B = {ssim_hsv_b}')

    pyplot.subplot(221)
    pyplot.imshow(image)
    pyplot.title('Original Image')
    pyplot.subplot(222)
    pyplot.imshow(clahe_rgb)
    pyplot.title(f'Contrast Enhanced Image by CLAHE')
    pyplot.subplot(223)
    pyplot.imshow(rgb_from_ycbcr)
    pyplot.title('Contrast Enhanced Image using YCbCr colour map')
    pyplot.subplot(224)
    pyplot.imshow(rgb_image_idwt_hsv)
    pyplot.title('Contrast Enhanced Image using HSV colour map')
    pyplot.show()

    return


def rgb2hsv(image):

    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    height, width = red_channel.shape

    c_max = numpy.maximum(red_channel, numpy.maximum(green_channel, blue_channel))
    c_min = numpy.minimum(red_channel, numpy.minimum(green_channel, blue_channel))
    difference = c_max - c_min

    s_saturation = numpy.zeros((height, width), dtype=float)
    h_hue = numpy.zeros((height, width), dtype=float)

    for i in range(height):
        for j in range(width):
            if c_max[i, j] != 0:
                s_saturation[i, j] = difference[i, j]/c_max[i, j]
            if difference[i, j] != 0:
                if c_max[i, j] == red_channel[i, j]:
                    h_hue[i, j] = 60 * (((green_channel[i, j] - blue_channel[i, j]) / difference[i, j]) % 6)
                elif c_max[i, j] == green_channel[i, j]:
                    h_hue[i, j] = 60 * (((blue_channel[i, j] - red_channel[i, j]) / difference[i, j]) + 2)
                else:
                    h_hue[i, j] = 60 * (((red_channel[i, j] - green_channel[i, j]) / difference[i, j]) + 4)

    v_luminannce = c_max

    hsv_image = numpy.zeros((height, width, 3), dtype=float)
    hsv_image[:, :, 0] = h_hue/360
    hsv_image[:, :, 1] = s_saturation
    hsv_image[:, :, 2] = v_luminannce

    return hsv_image


def hsv2rgb(hsv_image):

    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    v = hsv_image[:, :, 2]
    height, width = h.shape

    h = h * 360
    c = v*s    #chroma
    m = v - c        # rgb component with smallest value
    x = c*(1 - abs((h/60) % 2 - 1))     # an intermediate value used for computing the RGB model

    red_channel = numpy.zeros((height, width), dtype=float)
    green_channel = numpy.zeros((height, width), dtype=float)
    blue_channel = numpy.zeros((height, width), dtype=float)

    for i in range(height):
        for j in range(width):
            if 0 <= h[i, j] < 60:
                red_channel[i, j] = c[i, j]
                green_channel[i, j] = x[i, j]
                blue_channel[i, j] = 0
            if 60 <= h[i, j] < 120:
                red_channel[i, j] = x[i, j]
                green_channel[i, j] = c[i, j]
                blue_channel[i, j] = 0
            if 120 <= h[i, j] < 180:
                red_channel[i, j] = 0
                green_channel[i, j] = c[i, j]
                blue_channel[i, j] = x[i, j]
            if 180 <= h[i, j] < 240:
                red_channel[i, j] = 0
                green_channel[i, j] = x[i, j]
                blue_channel[i, j] = c[i, j]
            if 240 <= h[i, j] < 300:
                red_channel[i, j] = x[i, j]
                green_channel[i, j] = 0
                blue_channel[i, j] = c[i, j]
            if 300 <= h[i, j] < 360:
                red_channel[i, j] = c[i, j]
                green_channel[i, j] = 0
                blue_channel[i, j] = x[i, j]

    red_channel = red_channel + m
    green_channel = green_channel + m
    blue_channel = blue_channel + m
    image = numpy.zeros((height, width, 3), dtype=float)

    image[:, :, 0] = red_channel
    image[:, :, 1] = green_channel
    image[:, :, 2] = blue_channel

    return image


def clahe_(image):

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    clahe_image = clahe.apply((image*255).astype(numpy.uint8))

    return clahe_image


def clahe_on_rgb(image):

    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    height, width = red_channel.shape

    red_channel_clahe = clahe_(red_channel)
    green_channel_clahe = clahe_(green_channel)
    blue_channel_clahe = clahe_(blue_channel)

    clahe_image = numpy.zeros((height, width, 3), dtype=float)

    clahe_image[:, :, 0] = red_channel_clahe/255.0
    clahe_image[:, :, 1] = green_channel_clahe/255.0
    clahe_image[:, :, 2] = blue_channel_clahe/255.0

    return clahe_image


def rgb2ycbcr(image):

    image = (image*255).astype('int')
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    height, width = red_channel.shape

    y = 0.299*red_channel + 0.587*green_channel + 0.114*blue_channel
    cb = -0.169*red_channel - 0.331*green_channel + 0.499*blue_channel + 128
    cr = 0.499*red_channel - 0.418*green_channel -.0813*blue_channel + 128

    ycbcr_image = numpy.zeros((height, width, 3), dtype=float)
    ycbcr_image[:, :, 0] = y/255
    ycbcr_image[:, :, 1] = cb/255
    ycbcr_image[:, :, 2] = cr/255

    return ycbcr_image


def ycbcr2rgb(image):

    image = (image * 255).astype('int')
    y = image[:, :, 0]
    cb = image[:, :, 1]
    cr = image[:, :, 2]

    height, width = y.shape

    r = 1*y + 1.402*(cr - 128)
    g = y - 0.344*(cb - 128) - 0.714*(cr - 128)
    b = y + 1.772*(cb - 128)

    rgb_image = numpy.zeros((height, width, 3), dtype=float)
    rgb_image[:, :, 0] = r/255
    rgb_image[:, :, 1] = g/255
    rgb_image[:, :, 2] = b/255

    return rgb_image


def mse_(image1, image2):

    image1 = (image1*255).astype('int')
    image2 = (image2 * 255).astype('int')
    height, width = image1.shape
    diff = image1-image2
    sqr_diff = numpy.square(diff)
    sum_square = numpy.sum(sqr_diff)
    error = sum_square/(height*width)
    max_val = numpy.max(image1)
    psnr = 10*numpy.log10((max_val**2)/error)

    return [error, psnr]


def ssim_(image1, image2):

    image1 = (image1 * 255).astype('int')
    image2 = (image2 * 255).astype('int')
    height, width = image1.shape
    sigma_value = 1
    gauss_kernel_size = 3
    window = numpy.zeros((gauss_kernel_size, gauss_kernel_size), dtype=float)
    x = int(gauss_kernel_size / 2)
    y = int(gauss_kernel_size / 2)

    for m in range(-x, x + 1):
        for n in range(-y, y + 1):
            x1 = 2 * numpy.pi * (sigma_value ** 2)
            x2 = numpy.exp(-(m ** 2 + n ** 2) / (2 * sigma_value ** 2))
            window[m + x, n + y] = x2 / x1

    win_half = int(gauss_kernel_size / 2)
    mu_1 = numpy.zeros(image1.shape, dtype=float)
    mu_2 = numpy.zeros(image1.shape, dtype=float)
    variance_1 = numpy.zeros(image1.shape, dtype=float)
    variance_2 = numpy.zeros(image1.shape, dtype=float)
    covariance = numpy.zeros(image1.shape, dtype=float)
    luminance = numpy.zeros(image1.shape, dtype=float)
    contrast = numpy.zeros(image1.shape, dtype=float)
    structure = numpy.zeros(image1.shape, dtype=float)
    ssim = numpy.zeros(image1.shape, dtype=float)
    c1 = 0.01
    c2 = 0.01
    c3 = 0.01

    for i in range(win_half, height - win_half):
        for j in range(win_half, width - win_half):
            mu_1[i, j] = numpy.sum(numpy.multiply(window, image1[i-win_half:i+win_half+1, j-win_half:j+win_half+1]))
            mu_2[i, j] = numpy.sum(numpy.multiply(window, image2[i-win_half:i + win_half+1, j-win_half:j+win_half+1]))
            variance_1[i, j] = numpy.sum(numpy.multiply(window, (
                        image1[i-win_half:i + win_half+1, j-win_half:j+win_half+1] - mu_1[i, j])**2))
            variance_2[i, j] = numpy.sum(numpy.multiply(window, (
                        image2[i - win_half:i + win_half + 1, j - win_half:j + win_half + 1] - mu_2[i, j]) ** 2))
            covariance[i, j] = numpy.sum(numpy.multiply(window, numpy.multiply((
                        image1[i-win_half:i + win_half+1, j-win_half:j+win_half+1] - mu_1[i, j]), (
                        image2[i-win_half:i + win_half+1, j-win_half:j+win_half+1] - mu_2[i, j]))))
            luminance[i, j] = (2*mu_1[i, j]*mu_2[i, j] + c1)/(mu_1[i, j]**2 + mu_2[i, j]**2 + c1)
            contrast[i, j] = (2*numpy.sqrt(variance_1[i, j])*numpy.sqrt(variance_2[i,j]) + c2)/(variance_1[i, j] + variance_2[i, j] + c2)
            structure[i, j] = (covariance[i, j] + c3)/(numpy.sqrt(variance_1[i, j])*numpy.sqrt(variance_2[i,j]) + c3)
            ssim[i, j] = luminance[i,j]*contrast[i,j]*structure[i,j]
    ssim_12 = numpy.sum(ssim)/(height*width)

    return ssim_12


def main():

    contrast_enhancement()

    return


if __name__ == '__main__':
    main()
