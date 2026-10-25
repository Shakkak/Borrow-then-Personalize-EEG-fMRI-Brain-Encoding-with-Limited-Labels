"""Approximate 2-D scalp positions of THINGS-EEG2's 17 channels, for plots only.

The release keeps 17 occipital and parietal channels of a 64-channel
EASYCAP 10-10 layout. These are flat positions on a unit head seen
from above (nose up, x to the right), read off the standard 10-10 layout by
hand, close enough to draw where each channel sits, not for source analysis.
"""
from __future__ import annotations

POSITIONS = {
    "O1": (-0.28, -0.88), "Oz": (0.0, -0.92), "O2": (0.28, -0.88),
    "PO7": (-0.54, -0.74), "PO3": (-0.25, -0.67), "POz": (0.0, -0.69), "PO4": (0.25, -0.67), "PO8": (0.54, -0.74),
    "P7": (-0.70, -0.58), "P5": (-0.51, -0.53), "P3": (-0.34, -0.49), "P1": (-0.17, -0.47), "Pz": (0.0, -0.46),
    "P2": (0.17, -0.47), "P4": (0.34, -0.49), "P6": (0.51, -0.53), "P8": (0.70, -0.58),
}
