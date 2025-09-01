import numpy as np
import math

def _color_brew(n):
    """Generate n colors with equally spaced hues.
    From: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_export.py

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360.0 / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list

def _dynamic_float_formatter(t):
    if int(t) == t:
        return str(t)
    if t % 1 == 0:
        return str(t)
    if math.log10(abs(t)) >= 6 or math.log10(abs(t)) <= -4:
        return f"{t:.2e}"
    if math.log10(abs(t)) >= 2:
        return f"{t:.2f}".rstrip('0').rstrip('.')
    return f"{t:f}".rstrip('0').rstrip('.')