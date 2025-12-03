import loguru
import matplotlib.pyplot as plt
import pandas as pd

logger = loguru.logger


def abs_diff(
    df: pd.DataFrame, in_col_x: str, in_col_y: str, out_col: str
) -> pd.DataFrame:
    logger.info(f"Berechne die Absolute Differenz ({len(df)}) ...")
    df[out_col] = (df[in_col_x] - df[in_col_y]).abs()
    return df


def mean(df: pd.DataFrame, in_col: str, out_col: str) -> pd.DataFrame:
    logger.info(f"Berechne den Mittelwert ({len(df)}) ...")
    df[out_col] = df[in_col].mean()
    return df


def median(df: pd.DataFrame, in_col: str, out_col: str) -> pd.DataFrame:
    logger.info(f"Berechne den Median ({len(df)}) ...")
    df[out_col] = df[in_col].median()
    return df


def quantile(df: pd.DataFrame, in_col: str, value: float) -> float:
    logger.info(f"Berechnene das Quantile ({len(df)}) ...")
    return df[in_col].quantile(value)


def std_dev(df: pd.DataFrame, in_col: str, out_col: str) -> pd.DataFrame:
    logger.info(f"Berechne die Standardabweichung ({len(df)}) ...")
    df[out_col] = df[in_col].std()
    return df


if __name__ == "__main__":
    ddf = pl.DataFrame({"data": [0, 1, 1, 1, 4, 5, 5, 6, 9, 34]})

    ddf = mean(ddf, "data")
    ddf = std_dev(ddf, "data")
    ddf = z_score(ddf, "data")

    print(ddf)

    pdf = ddf.to_pandas()

    plt.hist(pdf["z_score"], bins="auto", alpha=0.7, color="steelblue")

    # Vertikale Linie bei Wert 5
    plt.axvline(3, color="red", linestyle="-.", linewidth=2)
    plt.axvline(-3, color="red", linestyle="-.", linewidth=2)
    plt.show()

    a = pl.DataFrame({"d": [x for x in range(1000)]})
    print(a.head(1000))
