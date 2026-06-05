import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from sklearn.metrics import r2_score, mean_squared_error


def biomass_scatter(ax, df, field_col, pred_col, sd_col, title):
    mask = df[field_col].notna() & df[pred_col].notna() & df[sd_col].notna()

    df = df.loc[mask].copy()

    xmin = min(df[field_col].min(), df[pred_col].min())

    xmax = max(df[field_col].max(), df[pred_col].max())

    # Error bars
    ax.errorbar(
        df[field_col],
        df[pred_col],
        yerr=df[sd_col],
        fmt="none",
        ecolor="gray",
        alpha=0.6,
        capsize=2,
        zorder=1,
    )

    # Scatter
    ax.scatter(df[field_col], df[pred_col], s=60, alpha=0.8, zorder=2)

    # 1:1 line
    ax.plot([xmin, xmax], [xmin, xmax], "--k", linewidth=2, label="1:1")

    # Regression
    slope, intercept, r, p, se = linregress(df[field_col], df[pred_col])

    xfit = np.linspace(xmin, xmax, 100)

    ax.plot(xfit,
            intercept + slope * xfit,
            color="red",
            linewidth=2,
            label="Linear Fit")

    # Bias line
    bias = (df[pred_col] - df[field_col]).mean()

    # ax.plot(
    #     xfit,
    #     xfit + bias,
    #     color="blue",
    #     linestyle=":",
    #     linewidth=2,
    #     label=f"1:1 + Bias"
    # )

    r2 = r2_score(df[field_col], df[pred_col])

    rmse = np.sqrt(mean_squared_error(df[field_col], df[pred_col]))

    stats = (f"N = {len(df)}\n"
             f"R² = {r2:.2f}\n"
             f"RMSE = {rmse:.1f}\n"
             f"Bias = {bias:.1f}\n"
             f"Slope = {slope:.2f}")

    ax.text(
        0.05,
        0.95,
        stats,
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.9),
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    ax.set_aspect("equal")

    ax.set_title(title)

    ax.set_xlabel(r"Field AGB (Mg ha$^{-1}$)")

    ax.set_ylabel(r"Product AGB (Mg ha$^{-1}$)")

    ax.legend(loc="upper right")
