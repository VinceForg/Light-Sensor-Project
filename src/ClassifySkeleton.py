"""Part 4 skeleton -- YOU write the lookup.

The tables, the grid, and the scaling are done.
Fill in the two TODOs, then check against classify_point.py.
"""

import numpy as np

from PipelineCommon import (
    load_split_scaled,
    scale_features,
    quantize,
    dequantize,
    GRID_N,
    REJECT_TAU
)


bounds = load_split_scaled()["bounds"]

surfaces = np.load(
    "surfaces.npz"
)

classes = sorted(
    int(k[1:])
    for k in surfaces.files
)


tables = {
    c: quantize(
        surfaces[f"c{c}"]
    )
    for c in classes
}


def classify(raw_point):
    """Class label for a RAW point, or None for unknown."""

    point, _ = scale_features(
        np.asarray(raw_point),
        bounds
    )


    # TODO 1:
    # Turn point into a grid cell.
    #
    # col = int(point[0] * GRID_N)
    # row = int(point[1] * GRID_N)
    #
    # Then clip both into [0, GRID_N - 1].

    col = int(
        np.clip(
            point[0] * GRID_N,
            0,
            GRID_N - 1
        )
    )

    row = int(
        np.clip(
            point[1] * GRID_N,
            0,
            GRID_N - 1
        )
    )


    # TODO 2:
    # Read tables[c][row, col] for every class,
    # convert with dequantize(), and return the class
    # with the largest value.
    #
    # If the largest value is below REJECT_TAU,
    # return None.

    values = np.array([
        dequantize(
            tables[c][row, col]
        )
        for c in classes
    ])

    best = int(
        np.argmax(values)
    )

    if values[best] < REJECT_TAU:
        return None

    return classes[best]


if __name__ == "__main__":

    # One raw point on each shape,
    # then the ring's hollow centre.
    for q in (
        [0.0, -1.8],
        [0.0, 5.0],
        [6.8, 1.6],
        [4.8, 1.6]
    ):

        print(
            q,
            "->",
            classify(q)
        )