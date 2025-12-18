import pandas as pd

from outlier_detection.columns import NAME_DATA, NAME_OTHER


def categorial_x_categorial() -> pd.DataFrame:
    data = (
        [["A", "V"]] * 1
        + [["B", "V"]] * 2
        + [["C", "V"]] * 10
        + [["D", "V"]] * 500
        + [["E", "V"]] * 1000
    )
    data += (
        [["A", "W"]] * 2
        + [["B", "W"]] * 4
        + [["C", "W"]] * 8
        + [["D", "W"]] * 1000
        + [["E", "W"]] * 2000
    )
    data += (
        [["A", "X"]] * 10
        + [["B", "X"]] * 20
        + [["C", "X"]] * 200
        + [["D", "X"]] * 5000
        + [["E", "X"]] * 10000
    )
    data += (
        [["A", "Y"]] * 500
        + [["B", "Y"]] * 1000
        + [["C", "Y"]] * 50000
        + [["D", "Y"]] * 250000
        + [["E", "Y"]] * 50000
    )
    data += (
        [["A", "Z"]] * 1000
        + [["B", "Z"]] * 2000
        + [["C", "Z"]] * 100000
        + [["D", "Z"]] * 500000
        + [["E", "Z"]] * 1000
    )
    pdf = pd.DataFrame(data, columns=[NAME_DATA, NAME_OTHER])
    return pdf


if __name__ == "__main__":
    categorial_x_categorial()
