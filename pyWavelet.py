import pywt
import numpy as np


def normalize(component):
    """
    Normalize a given component by subtracting the mean and dividing by the standard deviation.

    Parameters:
    - component: The array to be normalized.

    Returns:
    - The normalized component.
    """
    mean = np.mean(component)
    std = np.std(component)
    return (component - mean) / std


def pywt_swt2(X, level, wave_name):
    """
    Perform stationary wavelet transform (SWT) on a 2D signal and normalize the components.

    Parameters:
    - X: Input 2D signal.
    - level: The level of the wavelet transform.
    - wave_name: The name of the wavelet to be used.

    Returns:
    - wavelets: The concatenated array of normalized wavelet components.
    """
    # Perform 2D stationary wavelet transform (SWT) at the specified level
    coeffs = pywt.swt2(X, wave_name, level, trim_approx=True)
    wavelets = []

    # Normalize the first component
    first_component = pywt.swt2(X, wave_name, 1, trim_approx=True)[0]
    wavelets.append(normalize(first_component))

    # Normalize and combine subsequent components
    for i in range(1, len(coeffs)):
        # Compute the synthesized component by combining the horizontal, vertical, and diagonal coefficients
        coeffs_H = 0.3 * coeffs[i][0] + 0.4 * coeffs[i][1] + 0.3 * coeffs[i][2]

        # Normalize the synthesized component
        wavelets.append(normalize(coeffs_H))

    # Concatenate all normalized components
    wavelets = np.concatenate(wavelets, axis=0)
    return wavelets
