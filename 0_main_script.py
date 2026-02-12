import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
    Change me if your images are in a different folder.
    Set the path to the folder containing all image files.
    NOTE: Each set of images path defined at begin for each Q.
"""
# import os
# images_base_path = "../images"
# os.chdir(images_base_path)


""" ================================================================================================================    
                                        Q1: Point-wise and histogram
    ================================================================================================================ """

mri_spine_img_path = "mri_spine.jpg"


def plot_imgs_and_hist(source_image, output_image,
                       normalized_source_image=False, normalized_output_image=False,
                       source_image_title="Source Image", output_image_title="Output Image"):
    def compute_hist(img):
        return cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(source_image, cmap='gray')
    axes[0, 0].set_title(source_image_title)
    axes[0, 0].axis('off')

    hist_src = compute_hist(source_image)
    if normalized_source_image:
        hist_src = hist_src / hist_src.sum()
        axes[0, 1].set_title("Source Histogram (Normalized)")
        axes[0, 1].set_ylabel("Probability")
    else:
        axes[0, 1].set_title("Source Histogram (Non-Normalized)")
        axes[0, 1].set_ylabel("Pixel Count")
    axes[0, 1].plot(hist_src, color='black')
    axes[0, 1].set_xlim([0, 255])
    axes[0, 1].set_xlabel("Gray Level")
    axes[0, 1].grid(True)

    axes[1, 0].imshow(output_image, cmap='gray')
    axes[1, 0].set_title(output_image_title)
    axes[1, 0].axis('off')

    hist_out = compute_hist(output_image)
    if normalized_output_image:
        hist_out = hist_out / hist_out.sum()
        axes[1, 1].set_title("Output Histogram (Normalized)")
        axes[1, 1].set_ylabel("Probability")
    else:
        axes[1, 1].set_title("Output Histogram (Non-Normalized)")
        axes[1, 1].set_ylabel("Pixel Count")
    axes[1, 1].plot(hist_out, color='black')
    axes[1, 1].set_xlim([0, 255])
    axes[1, 1].set_xlabel("Gray Level")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


""" 1_1 """


def PowerLawTransformation():
    """
        Power-Law (Gamma) Transformation – Function Analysis
        This section analyzes the power-law function s = c * r^gamma
        for multiple gamma values, including normalization to [0,1] and scaling back to [0,255]

        gamma < 1 enhances dark regions
        gamma = 1 keeps image unchanged
        gamma > 1 suppresses dark regions and enhances bright regions
        c controls overall scaling (contrast gain)
    """

    # Define gamma values and constant c
    gamma_values = [0.04, 0.1, 0.2, 0.4, 0.67, 1, 1.5, 2.5, 5, 10, 25]
    c = 1.0
    # Normalized intensity range [0,1]
    r = np.linspace(0, 1, 256)
    plt.figure(figsize=(8, 6))
    for gamma in gamma_values:
        s = c * (r ** gamma)
        plt.plot(r, s, label=f"gamma={gamma}")

    plt.xlabel("Input intensity r (normalized)")
    plt.ylabel("Output intensity s (normalized)")
    plt.title("Power-Law Transformation Curves")
    plt.legend()
    plt.grid(True)
    plt.show()


""" 1_2 """


def PLTusingLUT():
    """
    Applying Power-Law Transformation Using LUT
    This section applies gamma correction to an image using a precomputed Lookup Table (LUT)

    gamma < 1 enhances dark regions
    gamma = 1 keeps image unchanged
    gamma > 1 suppresses dark regions and enhances bright regions
    c controls overall scaling (contrast gain)

    For image "mri_spine.jpg" we would like to enhance dark regions: c=1, gamma=0.67
    Histograms shows small differences
    """

    src_img = cv2.imread(mri_spine_img_path, cv2.IMREAD_GRAYSCALE)
    c = 1.0
    gamma = 0.67

    # DICTIONARY LOOKUP TABLE LUT
    # Precompute LUTs for all (gamma, c) pairs
    gamma_values = [0.04, 0.1, 0.2, 0.4, 0.67, 1, 1.5, 2.5, 5, 10, 25]
    c_values = [0.8, 1.0, 1.2]
    r = np.arange(256)
    r_norm = r / 255.0
    lut_dict = {}
    for gamma_i in gamma_values:
        for c_i in c_values:
            lut = np.clip(c_i * np.power(r_norm, gamma_i) * 255, 0, 255).astype(np.uint8)
            lut_dict[(gamma_i, c_i)] = lut

    # Gamma correction using lookup table
    gamma_img = cv2.LUT(src_img, lut_dict[(gamma, c)])
    plot_imgs_and_hist(src_img, gamma_img, True, True,
                       "Input image: mri_spine.jpg",
                       "Gamma corrected image using LUT c=1, gamma=0.67")


""" 1_3 """


def EqHist(img):
    """
    Histogram Equalization Using LUT
    :param img: np.ndarray Grayscale image
    :return: equalized histogram image along with eq histogram and LUT
    """

    hist, _ = np.histogram(img.flatten(), 256, range=(0, 256))
    cdf = hist.cumsum()  # Compute cumulative distribution function (CDF)
    cdf_normalized = cdf / cdf[-1]
    lut = np.floor(255 * cdf_normalized).astype(np.uint8)  # Create LUT
    equalized = cv2.LUT(img, lut)  # Apply LUT
    hist_eq, _ = np.histogram(equalized.flatten(), 256, range=(0, 256))
    return equalized, hist_eq, lut


def mriSpineEqHist_v1():
    """
        Histogram Equalization Using LUT
        The histogram equalization result for "mri_spine.jpg" is not uniform because the
        original histogram has a high concentration of pixels at gray level 0.
        This creates a large jump in the CDF, causing many pixels to map to the same
        output gray level, resulting in a peak in the equalized histogram.
    """

    img = cv2.imread(mri_spine_img_path, cv2.IMREAD_GRAYSCALE)
    hist_orig, _ = np.histogram(img.flatten(), 256, range=(0, 256))  # Compute original histogram
    eq_img, eq_hist, lut = EqHist(img)

    # Plot images and histograms
    plt.figure(figsize=(8, 10))
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax2.plot(hist_orig, color='black')
    ax2.set_title("Original Histogram")
    ax2.set_xlabel("Gray Level")
    ax2.set_ylabel("Pixel Count")
    ax2.set_xlim([0, 255])
    ax2.grid(True)

    # Row 2: equalized image and histogram
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax3.imshow(eq_img, cmap='gray', vmin=0, vmax=255)
    ax3.set_title("Equalized Image")
    ax3.axis('off')

    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax4.plot(eq_hist, color='black')
    ax4.set_title("Equalized Histogram")
    ax4.set_xlabel("Gray Level")
    ax4.set_ylabel("Pixel Count")
    ax4.set_xlim([0, 255])
    ax4.grid(True)

    # Row 3: LUT spanning both columns
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax5.plot(np.arange(256), lut, color='black')
    ax5.set_title("Histogram Equalization LUT")
    ax5.set_xlabel("Input Gray Level")
    ax5.set_ylabel("Output Gray Level")
    ax5.set_xlim([0, 255])
    ax5.set_ylim([0, 255])
    ax5.grid(True)

    plt.tight_layout()
    plt.show()


""" 1_4 """


def EqHist_v2(img, threshold):
    """
    Histogram Equalization Using LUT
    :param threshold: mask for img to apply on the operation
    :param img: np.ndarray Grayscale image
    :return: equalized histogram image along with eq histogram and LUT
    """

    img_copy = img.copy()
    mask = img_copy >= threshold  # Mask pixels above threshold
    selected_pixels = img_copy[mask]  # Extract relevant pixels
    hist, _ = np.histogram(selected_pixels.flatten(), 256, range=(0, 256))
    cdf = hist.cumsum()  # Compute cumulative distribution function (CDF)
    cdf_normalized = cdf / cdf[-1]
    lut = np.floor(255 * cdf_normalized).astype(np.uint8)  # Create LUT
    equalized = cv2.LUT(img, lut)  # Apply LUT
    hist_eq, _ = np.histogram(equalized.flatten(), 256, range=(0, 256))
    return equalized, hist_eq, lut


def mriSpineEqHist_v2():
    """
    Threshold Histogram Equalization (Selective HE)
    Performs histogram equalization only for pixels above a threshold T, pixels below T remain unchanged.

    T = 20 is chosen just above the large peak near gray level 0 in the source histogram,
    so background/very dark pixels remain unchanged while histogram equalization
    is applied only to the informative intensity range

    Compare to section C, this will not affect the dominate gray levels [0,20] which are the
    dark area.
    """

    img = cv2.imread(mri_spine_img_path, cv2.IMREAD_GRAYSCALE)
    hist_orig, _ = np.histogram(img.flatten(), 256, range=(0, 256))
    thershold = 20
    eq_img, eq_hist, lut = EqHist_v2(img, thershold)

    # Plot images and histograms
    plt.figure(figsize=(8, 10))
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax2.plot(hist_orig, color='black')
    ax2.set_title("Original Histogram")
    ax2.set_xlabel("Gray Level")
    ax2.set_ylabel("Pixel Count")
    ax2.set_xlim([0, 255])
    ax2.grid(True)

    # Row 2: equalized image and histogram
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax3.imshow(eq_img, cmap='gray', vmin=0, vmax=255)
    ax3.set_title("Equalized Image")
    ax3.axis('off')

    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax4.plot(eq_hist, color='black')
    ax4.set_title("Equalized Histogram")
    ax4.set_xlabel("Gray Level")
    ax4.set_ylabel("Pixel Count")
    ax4.set_xlim([0, 255])
    ax4.grid(True)

    # Row 3: LUT spanning both columns
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax5.plot(np.arange(256), lut, color='black')
    ax5.set_title("Histogram Equalization LUT")
    ax5.set_xlabel("Input Gray Level")
    ax5.set_ylabel("Output Gray Level")
    ax5.set_xlim([0, 255])
    ax5.set_ylim([0, 255])
    ax5.grid(True)

    plt.tight_layout()
    plt.show()


""" 1_5 """


def mriSpineEqHist_v3():
    """
    Block-wise Thresholded Histogram Equalization

    1. Divide the image manually into 9 blocks (not necessarily equal size).
    2. For each block, choose a threshold T manually based on the histogram.
    3. Apply thresholded HE (section D) per block.
    4. Combine blocks back into output image.

    Notes / Answers:
        - Yes, we can set different thresholds per block, because each block has different intensity distributions.
        - The chosen thresholds here are examples: darker blocks get lower T, brighter blocks higher T.
        - This improves contrast locally, especially when image has both very dark and very bright regions.
        - For very uniform images (all pixels nearly same intensity), no improvement is expected
            because thresholded HE will have almost no effect.
    """

    image = cv2.imread(mri_spine_img_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape

    # Define 9 blocks manually: separating dark vs bright areas
    # Each block is (y_start, y_end, x_start, x_end)
    blocks = [
        (0, h // 3, 0, w // 3),
        (0, h // 3, w // 3, 2 * w // 3),
        (0, h // 3, 2 * w // 3, w),
        (h // 3, 2 * h // 3, 0, w // 3),
        (h // 3, 2 * h // 3, w // 3, 2 * w // 3),
        (h // 3, 2 * h // 3, 2 * w // 3, w),
        (2 * h // 3, h, 0, w // 3),
        (2 * h // 3, h, w // 3, 2 * w // 3),
        (2 * h // 3, h, 2 * w // 3, w)
    ]

    # Thresholds per block (example: higher for bright regions, lower for dark)
    thresholds = [15, 12, 60, 35, 22, 10, 12, 13, 20]
    output = np.zeros_like(image)
    for idx, (y0, y1, x0, x1) in enumerate(blocks):
        block = image[y0:y1, x0:x1]
        T = thresholds[idx]
        mask = block >= T
        selected_pixels = block[mask]
        # Compute histogram of pixels above threshold
        hist, _ = np.histogram(selected_pixels, bins=256, range=(0, 256))
        cdf = hist.cumsum()
        if cdf[-1] == 0:  # Avoid division by zero
            lut = np.arange(256, dtype=np.uint8)
        else:
            cdf_normalized = cdf / cdf[-1]
            lut = np.floor(255 * cdf_normalized).astype(np.uint8)

        # Apply LUT only to pixels >= T
        eq_block = block.copy()
        eq_block[mask] = lut[block[mask]]
        output[y0:y1, x0:x1] = eq_block

    plot_imgs_and_hist(image, output, True, True)


# PowerLawTransformation()
# PLTusingLUT()
# mriSpineEqHist_v1()
# mriSpineEqHist_v2()
# mriSpineEqHist_v3()

""" ================================================================================================================    
                                        Q2: Frequency domain
    ================================================================================================================ """

Uma_img_path = "Uma.jpg"

""" 2_1 """


def Q2_a():
    """
        Load 'Uma.jpg' in grayscale, compute its 2D Fourier Transform,
        and display the original image along with its magnitude spectrum on a log scale
    """

    img = cv2.imread(Uma_img_path, cv2.IMREAD_GRAYSCALE)
    # Compute Fourier Transform (2D)
    f_transform = np.fft.fft2(img)
    # Shift zero frequency to center
    f_shifted = np.fft.fftshift(f_transform)
    # Calculate magnitude spectrum (log scale for better visibility)
    magnitude_spectrum = 20 * np.log(np.abs(f_shifted) + 1)

    plt.figure(figsize=(12, 6))
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')
    # Magnitude spectrum
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum (Log Scale)')
    plt.xlabel('Horizontal Frequency (u)')
    plt.ylabel('Vertical Frequency (v)')
    rows, cols = img.shape
    cx, cy = cols // 2, rows // 2
    plt.xticks([0, cx, cols], ['-High', '0', '+High'])
    plt.yticks([0, cy, rows], ['-High', '0', '+High'])
    plt.tight_layout()
    plt.show()


""" 2_2 """


def Q2_b():
    """
        Create a mask that preserves the 5% lowest frequencies in both the x and y directions,
        forming a cross shape in the shifted FFT spectrum. Then, apply the mask and perform
        the inverse Fourier Transform to reconstruct the image.
    """

    img = cv2.imread(Uma_img_path, cv2.IMREAD_GRAYSCALE)
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = f_shift.shape
    crow, ccol = rows // 2, cols // 2  # centers

    # Calculate strip widths for 5% of the frequencies
    h_width = max(1, int(0.05 * rows))
    w_width = max(1, int(0.05 * cols))

    # Create a cross-shaped mask (union of vertical and horizontal strips)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[crow - h_width // 2: crow + h_width // 2, :] = 1
    mask[:, ccol - w_width // 2: ccol + w_width // 2] = 1

    # Apply mask to shifted FFT
    f_shift_filtered = f_shift * mask

    # Reconstruct the image
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(20 * np.log(np.abs(f_shift_filtered) + 1), cmap='gray')
    plt.title('Filtered Spectrum (5% Low Freq Cross)')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img_back, cmap='gray')
    plt.title('Reconstructed Image (Part B)')
    plt.axis('off')
    plt.show()


""" 2_3, 2_4, 2_5 """


def Q2_c_d_e():
    """
    Identify the top 5% dominant rows and columns based on magnitude summation, create a mask from their union,
    and reconstruct the image using the inverse Fourier Transform.

        Q2_c:
            The dominant frequency columns are identified by summing the magnitude values along each column
            and selecting the top 5% with the highest energy.
        Q2_d:
            The dominant frequency rows are identified by summing the magnitude values along each row
            and selecting the top 5% with the highest energy.
        Q2_e:
            A binary mask is constructed to preserve only the dominant rows and columns in the frequency domain.
            The filtered spectrum is inverted back to the spatial domain, producing a reconstructed image
            that retains the main structural content while suppressing fine details.
    """

    img = cv2.imread(Uma_img_path, cv2.IMREAD_GRAYSCALE)
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = f_shift.shape
    magnitude = np.abs(f_shift)

    # Section (c): Find 5% dominant columns
    col_sums = np.sum(magnitude, axis=0)
    num_cols_to_keep = max(1, int(0.05 * cols))
    dominant_col_indices = np.argsort(col_sums)[-num_cols_to_keep:]

    # Section (d): Find 5% dominant rows
    row_sums = np.sum(magnitude, axis=1)
    num_rows_to_keep = max(1, int(0.05 * rows))
    dominant_row_indices = np.argsort(row_sums)[-num_rows_to_keep:]

    # Section (e): Create mask for dominant rows and columns
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[dominant_row_indices, :] = 1
    mask[:, dominant_col_indices] = 1

    # Apply mask
    f_shift_filtered = f_shift * mask

    # Reconstruct the image
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(20 * np.log(np.abs(f_shift_filtered) + 1), cmap='gray')
    plt.title('Filtered Spectrum (5% Dominant Rows/Cols)')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img_back, cmap='gray')
    plt.title('Reconstructed Image (Part E)')
    plt.axis('off')
    plt.show()


""" 2_6 """


def Q2_f():
    """
        Identify the top 10% most dominant individual frequencies in the 2D magnitude spectrum,
        apply a corresponding mask, and reconstruct the image using the inverse Fourier Transform.
    """

    img = cv2.imread(Uma_img_path, cv2.IMREAD_GRAYSCALE)
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = f_shift.shape
    magnitude = np.abs(f_shift)

    # Flatten magnitude to find the top 10% threshold
    flat_magnitude = magnitude.flatten()
    num_pixels = len(flat_magnitude)
    num_to_keep = max(1, int(0.10 * num_pixels))

    # Find the threshold value for the top 10%
    # We can use np.partition for efficiency
    threshold = np.sort(flat_magnitude)[-num_to_keep]

    # Create mask for frequencies with magnitude >= threshold
    mask = (magnitude >= threshold).astype(np.uint8)

    # Apply mask
    f_shift_filtered = f_shift * mask

    # Reconstruct the image
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(20 * np.log(np.abs(f_shift_filtered) + 1), cmap='gray')
    plt.title('Filtered Spectrum (Top 10% Dominant 2D frequencies)')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img_back, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')
    plt.show()


# Q2_a()
# Q2_b()
# Q2_c_d_e()
# Q2_f()


""" ================================================================================================================    
                                        Q3: Rotation and moving
    ================================================================================================================ """

cameraman_img_path = "cameraman.jpg"
Brad_img_path = "Brad.jpg"

""" 3_1 """


def shift_fractional_bilinear(image, dx, dy):
    """
        Shift image by fractional dx, dy in [0,1) using bilinear-interpolation
    """

    rows, cols = image.shape
    shifted = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            # Inverse mapping
            src_y = i - dy
            src_x = j - dx
            if 0 <= src_x < cols - 1 and 0 <= src_y < rows - 1:
                x0 = int(np.floor(src_x))
                y0 = int(np.floor(src_y))
                dx_f = src_x - x0
                dy_f = src_y - y0
                shifted[i, j] = \
                    (
                            (1 - dx_f) * (1 - dy_f) * image[y0, x0] +
                            dx_f * (1 - dy_f) * image[y0, x0 + 1] +
                            (1 - dx_f) * dy_f * image[y0 + 1, x0] +
                            dx_f * dy_f * image[y0 + 1, x0 + 1]
                    )

    return shifted


""" 3_2 """


def shift_image_bilinear(image, dx, dy):
    """
        Shift image by arbitrary (possibly non-integer) dx, dy
        using bilinear interpolation
    """

    # Separate integer and fractional parts
    dx_int = int(np.floor(dx))
    dy_int = int(np.floor(dy))
    dx_frac = dx - dx_int
    dy_frac = dy - dy_int
    rows, cols = image.shape
    shifted_int = np.zeros_like(image)
    # Integer shift (no interpolation)
    for i in range(rows):
        for j in range(cols):
            src_y = i - dy_int
            src_x = j - dx_int

            if 0 <= src_x < cols and 0 <= src_y < rows:
                shifted_int[i, j] = image[src_y, src_x]

    # Fractional shift
    return shift_fractional_bilinear(shifted_int, dx_frac, dy_frac)


""" 3_3 """


def CameramanShifting():
    # Load and normalize cameraman.jpg
    img_cameraman = cv2.imread(cameraman_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # Apply translation [dx, dy] = [170.3, 130.8]
    shift_x, shift_y = 170.3, 130.8
    img_translated = shift_image_bilinear(img_cameraman, shift_x, shift_y)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_cameraman, cmap='gray')
    plt.title("Original Cameraman")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img_translated, cmap='gray')
    plt.title(f"Translated (dx={shift_x}, dy={shift_y})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


""" 3_4 """


def BradMask1(plot=False):
    brad_path = 'Brad.jpg'
    img_brad = cv2.imread(Brad_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    rows, cols = img_brad.shape

    # Define the mask parameters
    center_y, center_x = 200, 250
    radius = 150

    # Create the binary mask (mask1): bottom half-circle
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask1 = (dist_from_center <= radius) & (y >= center_y)
    mask1 = mask1.astype(np.float32)

    # # Apply the mask to create brad_win
    # brad_win = img_brad * mask1

    if (plot):
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_brad, cmap='gray')
        plt.title("Original Brad.jpg")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(mask1, cmap='gray')
        plt.title("Bottom Half-Circle Mask (mask1)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return mask1


""" 3_5 """


def BradWin(plot=False):
    src = cv2.imread(Brad_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    brad_win = src * BradMask1(False)

    if (plot):
        plt.figure(figsize=(15, 6))
        plt.imshow(brad_win, cmap='gray')
        plt.title("BradWin")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return brad_win


""" 3_6 """


def Rotation(image, angle_deg, method='nearest'):
    """
        Rotates an image around its center using inverse mapping.
        Supports 'nearest' and 'bilinear' interpolation.
    """

    angle_rad = np.deg2rad(angle_deg)
    rows, cols = image.shape
    center_y, center_x = rows / 2.0, cols / 2.0

    # Create a grid of indices for the output image
    i_indices, j_indices = np.indices((rows, cols))

    # Shift indices so that the origin is at the center of the image
    x = j_indices - center_x
    y = i_indices - center_y

    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    # Perform inverse rotation to find source coordinates
    # R(-theta) = [[cos, sin], [-sin, cos]]
    src_x = x * cos_t + y * sin_t + center_x
    src_y = -x * sin_t + y * cos_t + center_y

    # Find pixels that map back into the source image boundaries
    valid_mask = (src_x >= 0) & (src_x < cols - 1) & (src_y >= 0) & (src_y < rows - 1)

    rotated = np.zeros_like(image)

    if method == 'nearest':
        # Nearest Neighbor interpolation
        src_x_near = np.round(src_x[valid_mask]).astype(int)
        src_y_near = np.round(src_y[valid_mask]).astype(int)
        rotated[valid_mask] = image[src_y_near, src_x_near]

    elif method == 'bilinear':
        # Bilinear interpolation
        x0 = np.floor(src_x[valid_mask]).astype(int)
        x1 = x0 + 1
        y0 = np.floor(src_y[valid_mask]).astype(int)
        y1 = y0 + 1

        dx = src_x[valid_mask] - x0
        dy = src_y[valid_mask] - y0

        # Weighted sum of 4 neighbors
        rotated[valid_mask] = (
                (1 - dx) * (1 - dy) * image[y0, x0] +
                dx * (1 - dy) * image[y0, x1] +
                (1 - dx) * dy * image[y1, x0] +
                dx * dy * image[y1, x1]
        )

    return rotated


def RotateBradWin(angles_deg, method, plot=False):
    results = {}
    brad_win = BradWin(False)
    # Perform rotations
    for angle in angles_deg:
        results[angle] = Rotation(brad_win, angle, method=method)

    if plot:
        plt.subplot(1, len(angles_deg), 1)
        plt.title("Rotation Comparison: Nearest Neighbor vs. Bilinear Interpolation")
        plt.axis('off')
        for i, angle in enumerate(angles_deg):
            plt.subplot(1, len(angles_deg), i + 1)
            plt.imshow(results[angle], cmap='gray')
            plt.title(f"{method.capitalize()} - {angle}°")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    return results


""" 3_7 """


def PlotRotationBrad():
    def crop_nonblack(image):
        """
        Crop image by removing fully black rows and columns.
        Keeps the aspect ratio of the remaining image.
        """
        # Find all rows and columns that are not completely black
        rows_nonzero = np.any(image > 0, axis=1)
        cols_nonzero = np.any(image > 0, axis=0)
        # Find the bounding box
        if not np.any(rows_nonzero) or not np.any(cols_nonzero):
            # Image is completely black
            return image

        row_start, row_end = np.where(rows_nonzero)[0][[0, -1]]
        col_start, col_end = np.where(cols_nonzero)[0][[0, -1]]
        # Crop the image
        return image[row_start:row_end + 1, col_start:col_end + 1]

    angles = [45, 60, 90]
    methods = ['nearest', 'bilinear']
    results = {}
    # Perform rotations by methods
    for method in methods:
        results[method] = RotateBradWin(angles, method, plot=False)

    # Plot, title: "Rotation Comparison: Nearest Neighbor vs. Bilinear Interpolation"
    # Plotting
    fig, axes = plt.subplots(len(methods), len(angles), figsize=(12, 8))
    plt.suptitle("Rotation Comparison: Nearest Neighbor vs. Bilinear Interpolation", fontsize=20)

    for i, method in enumerate(methods):
        for j, angle in enumerate(angles):
            res = crop_nonblack(results[method][angle])
            axes[i, j].imshow(res, cmap='gray')
            axes[i, j].set_title(f"{method.capitalize()} - {angle}°")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


# CameramanShifting()
# BradMask1(plot=True)
# BradWin(plot=True)
# PlotRotationBrad()


""" ================================================================================================================ 
                                        Q4: Edge Sharpening and Noise Analysis
                                NOTE: I used different image, "Inigo.jpg" does not exists!
     =============================================================================================================== """

heisenberg_img_path = "heisenberg.jpg"

""" Helpers: Laplacian, Salt & Paper noise, Median, Poisson noise """


def LaplacianSharpen(image, a):
    """
    NOTE: Using cv2.filter2D for convolution
    Laplacian kernel **2: [[0, 1,  0],
                           [1, -4, 1],
                           [0, 1,  0]]
    :param a: Argument for sharpening.
              Increasing enhances edges and details *but also increases high-frequency contrast
    :param image: [0,255]
    :return: delta - a * Laplacian**2
    """
    image_f = image.astype(np.float32)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap_filtered = cv2.filter2D(image_f, -1, laplacian_kernel)
    sharpened = image_f - a * lap_filtered
    if np.issubdtype(image.dtype, np.integer):
        return np.clip(sharpened, 0, 255).astype(image.dtype)
    else:
        return np.clip(sharpened, 0, 255)


def SAPnoise(image, prob=0.04):
    noisy = image.copy()
    rdn = np.random.random(image.shape)
    if np.issubdtype(image.dtype, np.integer):
        salt_val = 255
        pepper_val = 0
    else:
        salt_val = 255.0
        pepper_val = 0.0

    noisy[rdn > 1 - prob / 2] = salt_val
    noisy[rdn < prob / 2] = pepper_val
    return noisy


def DenoiseUsingMedain(image, ksize=3):
    """
    Applies median filtering to reduce salt-and-pepper noise in an image.
    :param image: grayscale image in range [0, 255]
    :param ksize: int, size of the median filter kernel (must be odd, e.g., 3, 5, 7)
    :return: denoised: np.ndarray, median-filtered image, same dtype as input
    """

    if not (np.issubdtype(image.dtype, np.integer) or np.issubdtype(image.dtype, np.floating)):
        raise ValueError("Image dtype must be integer or float32/float64")

    denoised = cv2.medianBlur(image, ksize)
    return denoised


def PoissonNoise(image):
    image_f = image.astype(np.float32) / 255.0
    noisy = np.random.poisson(image_f * 255.0) / 255.0  # Poisson, scale back to [0,1]
    return np.clip(noisy * 255, 0, 255).astype(image.dtype)


""" 4_1 """


def heisenbergLaplacianSharpen():
    """
    1.  Increasing 'a' enhances edges and details, but also increases high-frequency contrast.
    2.  The Laplacian filter sharpens edges because it is a second-order derivative operator
        that responds strongly to rapid intensity changes.
    3.  The parameter a controls the strength of this effect: small values produce mild sharpening, while larger
        values amplify edges more strongly but may introduce artifacts such as overshoot or clipping.
    4. image_f - a * lap_filtered:
        In flat areas of the image, lap_filtered ≈ 0, so the output remains almost identical to the original image.
        At edges, the Laplacian has large positive or negative values depending on the direction
        of the intensity change. Subtracting this term increases pixel values on the bright side of an edge
        and decreases them on the dark side, thereby increasing local contrast across the edge.
    5.  Chosen argument a = 0.25 is the best results
    """

    heisenberg_img = cv2.imread(heisenberg_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    a_values = [0.25, 0.45, 0.82, 1.0]

    plt.figure(figsize=(22, 8))
    plt.subplot(1, 5, 1)
    plt.imshow(heisenberg_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Original")
    plt.axis('off')
    for i, a in enumerate(a_values):
        sharpened_img = LaplacianSharpen(heisenberg_img, a)
        plt.subplot(1, 5, i + 2)
        plt.imshow(sharpened_img, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Sharpened (a={a})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


""" 4_2 """


def heisenbergLaplacianSharpen_A_above_1():
    """
        Increasing 'a' out of range [0,1] create noises, and high-frequency contrast
        This definitely create undesired results
    """

    heisenberg_img = cv2.imread(heisenberg_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    a_values = [1.3, 5]

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(heisenberg_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Original")
    plt.axis('off')
    for i, a in enumerate(a_values):
        sharpened_img = LaplacianSharpen(heisenberg_img, a)
        plt.subplot(1, 3, i + 2)
        plt.imshow(sharpened_img, cmap='gray')
        plt.title(f"Sharpened (a={a})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


""" 4_3 """


def HighContrastSAP():
    """
        Sharpening Salt & Pepper noise: The Laplacian filter dramatically amplifies the noise peaks
    """

    heisenberg_img = cv2.imread(heisenberg_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    img_salt_and_paper = SAPnoise(heisenberg_img, 0.04)
    a_values = [0.2, 0.7]

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(heisenberg_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Original")
    plt.axis('off')
    for i, a in enumerate(a_values):
        sharpened_img = LaplacianSharpen(img_salt_and_paper, a)
        plt.subplot(1, 3, i + 2)
        plt.imshow(sharpened_img, cmap='gray')
        plt.title(f"Sharpened (a={a})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


""" 4_4 """


def DenoiseSAP():
    """
        Sharpening Salt & Pepper noise using Median filtering and afterward Laplacian sharpen
    """

    heisenberg_img = cv2.imread(heisenberg_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    noisy_image = SAPnoise(heisenberg_img, 0.04)

    # Using Median filtering
    denoised = DenoiseUsingMedain(noisy_image, ksize=3)
    a_values = [0.2, 0.7]

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 4, 1)
    plt.imshow(heisenberg_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Noisy (Salt And Paper 4%)")
    plt.axis('off')

    for i, a in enumerate(a_values):
        sharpened = LaplacianSharpen(denoised, a)
        plt.subplot(1, 4, i + 3)
        plt.imshow(sharpened, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Sharpened (a={a})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


""" 4_5 """


def DenoisePoisson():
    heisenberg_img = cv2.imread(heisenberg_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    noisy_image = PoissonNoise(heisenberg_img)

    a_values = [0.2, 0.7]

    plt.figure(figsize=(12, 8))

    # Row 1: Noisy image + Laplacian sharpening
    plt.subplot(2, 3, 1)
    plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Noisy (Poisson)")
    plt.axis('off')

    for i, a in enumerate(a_values):
        sharpened = LaplacianSharpen(noisy_image, a)
        plt.subplot(2, 3, i + 2)
        plt.imshow(sharpened, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Laplacian (a={a})")
        plt.axis('off')

    # Row 2: Median + Laplacian
    denoised = DenoiseUsingMedain(noisy_image, ksize=3)

    plt.subplot(2, 3, 4)
    plt.imshow(denoised, cmap='gray', vmin=0, vmax=255)
    plt.title("Median filtered")
    plt.axis('off')

    for i, a in enumerate(a_values):
        sharpened = LaplacianSharpen(denoised, a)
        plt.subplot(2, 3, i + 5)
        plt.imshow(sharpened, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Median + Laplacian (a={a})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# heisenbergLaplacianSharpen()
# heisenbergLaplacianSharpen_A_above_1()
# HighContrastSAP()
# DenoiseSAP()
# DenoisePoisson()


""" ================================================================================================================ 
                                    Section 5: Filtering and morphological operations
     =============================================================================================================== """

crazyBioComp_img_path = "crazyBioComp.jpg"
keyboard_img_path = "keyboard.jpg"

""" 5_1 """


def crazyBioComp():
    crazyBioComp_img = cv2.imread(crazyBioComp_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Structuring elements
    kernel_square = np.ones((5, 5), np.uint8)
    kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    s_elements = [kernel_square, kernel_circle, kernel_cross]
    s_names = ['Square', 'Circle', 'Cross']

    # Morphological operations
    morph_ops = {
        'Erosion': cv2.erode,
        'Dilation': cv2.dilate,
        'Opening': lambda img, k: cv2.morphologyEx(img, cv2.MORPH_OPEN, k),
        'Closing': lambda img, k: cv2.morphologyEx(img, cv2.MORPH_CLOSE, k),
        'Top-Hat': lambda img, k: cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k),
        'Bottom-Hat': lambda img, k: cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k)
    }

    num_ops = len(morph_ops)
    num_kernels = len(s_elements)
    plt.figure(figsize=(5 * num_kernels, 4 * num_ops))
    idx = 1
    for op_name, op_func in morph_ops.items():
        for k_name, k in zip(s_names, s_elements):
            plt.subplot(num_ops, num_kernels, idx)
            result = op_func(crazyBioComp_img, k)
            combined = np.hstack((crazyBioComp_img, result))  # Original + processed
            plt.imshow(combined, cmap='gray', vmin=0, vmax=255)
            plt.title(f"{op_name}\n{k_name}")
            plt.axis('off')
            idx += 1

    plt.tight_layout()
    plt.show()


""" 5_2, 5_3, 5_4, 5_5, 5_6, 5_7, 5_8 """


def KeyboardCharsIsolation():
    keyboard_img = cv2.imread(keyboard_img_path, cv2.IMREAD_GRAYSCALE)
    normalized_img = keyboard_img.astype(np.float32) / 255.0

    # Create structuring elements
    kernel_v = np.zeros((8, 1), np.uint8)
    kernel_v[:, 0] = 1
    kernel_h = np.zeros((1, 8), np.uint8)
    kernel_h[0, :] = 1
    kernel_sq = np.ones((8, 8), np.uint8)

    # Apply erosion using vertical and horizontal line elements
    eroded_v = cv2.erode(normalized_img, kernel_v)
    eroded_h = cv2.erode(normalized_img, kernel_h)

    # Sum the two eroded results, threshold at 0.2, and invert
    sum_eroded = eroded_v + eroded_h
    _, binary_mask = cv2.threshold(sum_eroded, 0.2, 1.0, cv2.THRESH_BINARY)
    inverted_mask = (1.0 - binary_mask).astype(np.float32)

    # Apply median filter with 8x8 kernel
    # OpenCV's medianBlur requires uint8 input and an odd kernel size.
    # Using 7x7 or 9x9 as close approximations for 8x8, or manual implementation.
    # I used 7x7 as a standard odd size.
    inverted_mask_u8 = (inverted_mask * 255).astype(np.uint8)
    median_filtered = cv2.medianBlur(inverted_mask_u8, 7)

    # Refine mask by erosion with 8x8 square
    refined_mask_u8 = cv2.erode(median_filtered, kernel_sq)
    refined_mask = refined_mask_u8.astype(np.float32) / 255.0

    # Mask the original image
    _, refined_mask_bin = cv2.threshold(refined_mask, 0.5, 1.0, cv2.THRESH_BINARY)
    masked_img = normalized_img * refined_mask_bin

    # Sharpening
    sharpened_masked = LaplacianSharpen(masked_img, 0.2)

    # Final cleanup threshold to remove background bleed
    _, final_isolated = cv2.threshold(sharpened_masked, 0.1, 1.0, cv2.THRESH_BINARY)
    final_isolated_display = masked_img * final_isolated

    plt.figure(figsize=(20, 15))
    plt.subplot(2, 4, 1)
    plt.imshow(normalized_img, cmap='gray')
    plt.title("Normalized Original")
    plt.subplot(2, 4, 2)
    plt.imshow(eroded_v, cmap='gray')
    plt.title("Vertical Erosion")
    plt.subplot(2, 4, 3)
    plt.imshow(eroded_h, cmap='gray')
    plt.title("Horizontal Erosion")
    plt.subplot(2, 4, 4)
    plt.imshow(inverted_mask, cmap='gray')
    plt.title("Inverted Binary Mask")
    plt.subplot(2, 4, 5)
    plt.imshow(median_filtered, cmap='gray')
    plt.title("Median Filtered Mask")
    plt.subplot(2, 4, 6)
    plt.imshow(refined_mask_bin, cmap='gray')
    plt.title("Refined Square Mask")
    plt.subplot(2, 4, 7)
    plt.imshow(masked_img, cmap='gray')
    plt.title("Masked Result")
    plt.subplot(2, 4, 8)
    plt.imshow(final_isolated_display, cmap='gray')
    plt.title("Final Isolated Keys")
    plt.tight_layout()
    plt.show()


# crazyBioComp()
# KeyboardCharsIsolation()


""" ================================================================================================================ 
                                        Q6: Template Matching
     =============================================================================================================== """


Text_img_path = "Text.jpg"
font_c_img_path = "c.jpg"
font_k_img_path = "k.jpg"
font_E_sizes = [10, 11, 12, 14, 16]
font_E_templates_img = [cv2.imread(f'E{size}.jpg', cv2.IMREAD_GRAYSCALE) for size in font_E_sizes]


""" 6_1 """


def TemplateMatchingSSD(I, T, invalid_value=1e12):
    """
     S[x,y] = SUM_{i=1 to M} SUM_{j=1 to N}
                  ( T[i, j] - I[x + i - M/2 - 1, y + j - N/2 - 1] )**2
    :param I: np.ndarray (H,W) Greyscale image
    :param T: np.ndarray (h,w) Greyscale template
    :param invalid_value: float return value where is no valid matching in edges of img
    :return: SSD template matching in img
    """

    H, W = I.shape
    M, N = T.shape
    S = np.full((H, W), invalid_value, dtype=np.float64)
    half_M = M // 2
    half_N = N // 2

    for x in range(H):
        for y in range(W):
            ssd = 0.0
            valid = True

            for i in range(M):  # i = 0 .. M-1  =>  no need to sub 1
                for j in range(N):  # j = 0 .. N-1  =>  no need to sub 1
                    Ix = x + i - half_M
                    Iy = y + j - half_N
                    # Check image boundaries
                    if Ix < 0 or Ix >= H or Iy < 0 or Iy >= W:
                        valid = False
                        break

                    diff = float(T[i, j]) - float(I[Ix, Iy])
                    ssd += diff * diff

                if not valid:
                    break

            if valid:
                S[x, y] = ssd

    return S


""" 6_2 """


def EdgarAllanPoeText():
    """
        Find the font size in image 'Text.jpg'
        The font size is 10 (E10 has the best SSD result of template matching)
    """

    dict_results = {}
    img_text_raw = cv2.imread(Text_img_path, cv2.IMREAD_GRAYSCALE)
    normalized_img_text = img_text_raw.astype(np.float64) / 255.0

    # Compute SSD for each font size
    for font_size, font_template in zip(font_E_sizes, font_E_templates_img):
        ssd_matrix = TemplateMatchingSSD(normalized_img_text.astype(np.float32), font_template.astype(np.float32))
        # Look for the min value in SSD matrix which is the best matching (the image keeps font size as const size)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ssd_matrix)
        dict_results[font_size] = min_val

    # Identify best matching font size (lowest SSD)
    best_size = min(dict_results, key=dict_results.get)
    print(f"Identified Font Size: {best_size}")

    plt.figure(figsize=(15, 3))
    for i, size in enumerate(font_E_sizes):
        plt.subplot(1, len(font_E_sizes), i + 1)
        plt.imshow(font_E_templates_img[i], cmap='gray')
        plt.title(f"Size {size}\nSSD: {dict_results[size]:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


""" 6_3 """


def CountMatchLetter(image, template, threshold):
    res = TemplateMatchingSSD(image.astype(np.float32), template.astype(np.float32))
    loc = np.where(res <= threshold)

    # Non-maximum suppression to avoid multiple counts for the same character
    peaks = []
    if len(loc[0]) > 0:
        # Sort by match quality
        coords = list(zip(loc[0], loc[1]))
        coords.sort(key=lambda x: res[x[0], x[1]])

        h_t, w_t = template.shape
        while coords:
            y, x = coords[0]
            peaks.append((y, x))
            # Remove neighbors
            coords = [c for c in coords if abs(c[0] - y) > h_t / 2 or abs(c[1] - x) > w_t / 2]

    return len(peaks), res, peaks


def Count_a_t_letters():
    """
    Character counts:
        - 'a': 29
        - 'A': 3
        - 't': 26
        - 'T': 3
    """

    img_text_raw = cv2.imread(Text_img_path, cv2.IMREAD_GRAYSCALE)
    normalized_img_text = img_text_raw.astype(np.float32) / 255.0

    # Find representative template manually
    char_templates = \
        {
            'a': normalized_img_text[14:26, 55:64].astype(np.float32),
            'A': normalized_img_text[326:340, 63:73].astype(np.float32),
            't': normalized_img_text[12:26, 186:193].astype(np.float32),
            'T': normalized_img_text[117:131, 15:25].astype(np.float32)
        }

    threshold = 2
    counts = {}
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 4, 2)
    plt.imshow(img_text_raw, cmap='gray')
    plt.title('Text Image Source')
    for i, (char_st, template) in enumerate(char_templates.items()):
        count, score_map, locs = CountMatchLetter(normalized_img_text, template, threshold)
        counts[char_st] = count
        plt.subplot(2, 4, i + 5)
        plt.imshow(template, cmap='gray')
        plt.title(f"Template '{char_st}' (Found: {count})")

    plt.tight_layout()
    plt.show()

    print(f"Character counts:")
    for char_st, count in counts.items():
        print(f"- '{char_st}': {count}")


""" 6_4 """


def MrSmokeTooMuch():
    img_text_raw = cv2.imread(Text_img_path, cv2.IMREAD_GRAYSCALE)
    normalized_img_text = img_text_raw.astype(np.float32) / 255.0
    char_templates = \
        {
            'c': cv2.imread(font_c_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0,
            'k': cv2.imread(font_k_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        }

    threshold = 2
    _, _, coords = CountMatchLetter(normalized_img_text, char_templates['c'], threshold)

    # Action - replace all 'k' instances
    output_img = np.copy(normalized_img_text)
    h_c, w_c = char_templates['c'].shape
    for y, x in coords:
        # Coords are approximately should be center to matching location
        x -= w_c // 2
        y -= h_c // 2

        # Clear the 'c' area (fill with white)
        end_y_c = min(y + h_c, output_img.shape[0])
        end_x_c = min(x + w_c, output_img.shape[1])
        output_img[y:end_y_c, x:end_x_c] = 1.0

        # Calculate placement for 'k' centered in the original 'c' area
        # If sizes differ, center 'k' inside 'c' bounding box
        end_y_k = min(y + h_c, output_img.shape[0])
        end_x_k = min(x + w_c, output_img.shape[1])

        # Paste 'k' template into the output image
        output_img[y:end_y_k, x:end_x_k] = char_templates['k'][:h_c, :w_c]

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img_text_raw, cmap='gray')
    plt.title('Text Image Source')
    plt.subplot(1, 2, 2)
    plt.imshow(output_img, cmap='gray')
    plt.title("Replaced Character 'c' with 'k'")
    plt.tight_layout()
    plt.show()


# EdgarAllanPoeText()
# Count_a_t_letters()
# MrSmokeTooMuch()


""" ================================================================================================================ """
