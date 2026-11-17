
from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------- #
# Stage 1 : Gaussian smoothing filter  (Eq. 1, Eq. 2)
# --------------------------------------------------------------------------- #

def gaussian_kernel(ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Build the normalised smoothing kernel.

    Eq. (1):  w(i, j) = 1 / (2*pi*sigma^2)
                        * exp( -[ (i-x)^2 / (2*sigma^2) + (j-y)^2 / (2*sigma^2) ] )

              where (x, y) is the window centre. Working in centre-relative
              coordinates di = i - x, dj = j - y gives the isotropic Gaussian
              below. The leading 1/(2*pi*sigma^2) factor cancels during
              normalisation, but it is kept here to match the paper verbatim.

    Eq. (2):  k(i, j) = w(i, j) / sum_{i,j} w(i, j)     (weights sum to 1)

    The paper uses a 5x5 window (ksize = 5). It does not state a value for
    sigma, so it is exposed as a parameter (default 1.0, a common choice for
    a 5x5 kernel).
    """
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd (the window needs a centre pixel).")
    r = ksize // 2
    ax = np.arange(-r, r + 1)
    dj, di = np.meshgrid(ax, ax)                       # di rows, dj cols
    w = (1.0 / (2.0 * np.pi * sigma ** 2)) \
        * np.exp(-(di ** 2 + dj ** 2) / (2.0 * sigma ** 2))   # Eq. (1)
    k = w / w.sum()                                    # Eq. (2)
    return k


def gaussian_smooth(image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Apply the Gaussian smoothing filter of Eq. (1)-(2).

    The paper's procedure: align the filter centroid with the current pixel,
    multiply the covered pixels by the corresponding Gaussian weights, and sum
    the products to get the filtered value. That is exactly a 2-D convolution
    with the kernel from `gaussian_kernel`, implemented here with reflect
    padding so border pixels are handled without introducing dark edges.

    Returns a float64 image (not yet quantised); Stage 2 rounds it.
    """
    k = gaussian_kernel(ksize, sigma)
    r = ksize // 2
    img = image.astype(np.float64)
    padded = np.pad(img, r, mode="reflect")
    out = np.zeros_like(img)
    h, w = img.shape
    for a in range(ksize):
        for b in range(ksize):
            out += k[a, b] * padded[a:a + h, b:b + w]   # weighted sum of neighbours
    return out


# --------------------------------------------------------------------------- #
# Stage 2 : Histogram equalization  (Eq. 3, Eq. 4)
# --------------------------------------------------------------------------- #

def histogram_equalization(image: np.ndarray,
                           L: int = 256,
                           paper_denominator: bool = True) -> np.ndarray:
    """
    Flatten the intensity histogram of `image` (Eq. 3 - Eq. 4).

    Eq. (3):  CDF(a) = sum_{j=0}^{a} P(j)
              where P(j) is the count of pixels with grayscale value j. (The
              paper describes normalising the histogram; we keep raw counts so
              that the M*N term in Eq. (4) is dimensionally consistent - the two
              formulations are equivalent up to the M*N scaling.)

    Eq. (4):  H(a) = round( (CDF(a) - CDF(min)) / (M*N - 1) * (L - 1) )
              min = the minimum pixel value present in the image, so CDF(min)
              is the count of the darkest occurring intensity.

    `paper_denominator`:
        True  -> use (M*N - 1)          exactly as printed in Eq. (4).
        False -> use (M*N - CDF(min))   the canonical textbook denominator,
                 which guarantees the brightest bin maps to exactly L-1.
    """
    img = np.clip(np.round(image), 0, L - 1).astype(np.int64)
    M, N = img.shape

    hist = np.bincount(img.ravel(), minlength=L)        # P(j), raw counts
    cdf = np.cumsum(hist)                                # Eq. (3)

    a_min = int(np.nonzero(hist)[0][0])                 # minimum present value
    cdf_min = cdf[a_min]

    denom = (M * N - 1) if paper_denominator else (M * N - cdf_min)
    if denom <= 0:                                       # degenerate (flat) image
        return img.astype(np.uint8)

    H = np.round((cdf - cdf_min) / denom * (L - 1))      # Eq. (4)
    H = np.clip(H, 0, L - 1).astype(np.uint8)            # lookup table
    return H[img]                                        # remap every pixel


# --------------------------------------------------------------------------- #
# Full FSE pipeline
# --------------------------------------------------------------------------- #

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Angiography frames are grayscale; collapse an RGB(A) input if given."""
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        rgb = image[..., :3].astype(np.float64)
        # ITU-R BT.601 luma
        return rgb @ np.array([0.299, 0.587, 0.114])
    raise ValueError(f"Unsupported image shape {image.shape}")


def fse(image: np.ndarray,
        ksize: int = 5,
        sigma: float = 1.0,
        L: int = 256,
        paper_denominator: bool = True) -> np.ndarray:
    """
    Run the complete FSE enhancement: Gaussian smoothing (Eq. 1-2) followed by
    histogram equalization (Eq. 3-4).

    Parameters
    ----------
    image : 2-D grayscale or 3-D RGB(A) array (any numeric dtype).
    ksize : smoothing window size (paper: 5).
    sigma : Gaussian standard deviation (paper unspecified; default 1.0).
    L     : number of grayscale levels (paper: 8-bit imagery -> 256).
    paper_denominator : see `histogram_equalization`.

    Returns
    -------
    uint8 array in [0, L-1] with the same H x W as the input.
    """
    gray = to_grayscale(image)
    smoothed = gaussian_smooth(gray, ksize=ksize, sigma=sigma)     # Stage 1
    enhanced = histogram_equalization(smoothed, L=L,
                                      paper_denominator=paper_denominator)  # Stage 2
    return enhanced


# --------------------------------------------------------------------------- #
# Image-quality metric used by the paper  (Eq. 13)
# --------------------------------------------------------------------------- #

def image_std(image: np.ndarray, L: int = 256) -> float:
    """
    Normalised image standard deviation, sqrt(D(X)) with D(X) from Eq. (13),
    computed on intensities scaled to [0, 1]. The paper reports this value (S)
    to argue that a higher post-enhancement std means better contrast/quality
    (Tables 4-5; FSE reaches the largest S).
    """
    x = to_grayscale(image).astype(np.float64) / (L - 1)
    return float(np.sqrt(np.mean(x ** 2) - np.mean(x) ** 2))       # Eq. (13)


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #

def _main() -> None:
    import argparse
    from PIL import Image

    p = argparse.ArgumentParser(description="FSE image enhancement (Chen et al., 2025).")
    p.add_argument("input", help="path to an input angiography image")
    p.add_argument("output", help="path to write the enhanced image")
    p.add_argument("--ksize", type=int, default=5, help="smoothing window (default 5)")
    p.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma (default 1.0)")
    p.add_argument("--levels", type=int, default=256, help="grayscale levels L (default 256)")
    p.add_argument("--std-denominator", action="store_true",
                   help="use canonical (M*N - CDF_min) denominator instead of the paper's (M*N - 1)")
    args = p.parse_args()

    img = np.asarray(Image.open(args.input))
    out = fse(img, ksize=args.ksize, sigma=args.sigma, L=args.levels,
              paper_denominator=not args.std_denominator)

    Image.fromarray(out).save(args.output)
    print(f"input  S = {image_std(img, args.levels):.4f}")
    print(f"output S = {image_std(out, args.levels):.4f}")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    _main()