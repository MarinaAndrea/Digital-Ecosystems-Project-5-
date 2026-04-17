#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "dataset digiecos.xlsx"
OUTPUT_FILE = BASE_DIR / "technology_deployment_by_instrument.png"


# ============================================================
# CONFIG
# ============================================================

INSTRUMENT_ORDER = ["PPA", "Equity", "Development partnership"]
TECH_ORDER = ["Solar", "Wind", "Storage", "Nuclear", "Geothermal"]

COLORS = {
    "PPA": "#F2A317",
    "Equity": "#4F83C2",
    "Development partnership": "#A950B5",
}

TITLE = "Technology Deployment by Instrument"


# ============================================================
# HELPERS
# ============================================================

def normalize_column_name(col: object) -> str:
    return str(col).strip().strip('"').strip("'")


def extract_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"(20\d{2})", str(value))
    return int(match.group(1)) if match else None


def classify_instrument(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    lower = text.lower()

    if "aggregated stat" in lower:
        return "PPA"

    # Equity
    if "direct investment" in lower:
        return "Equity"
    if "equity" in lower:
        return "Equity"
    if "lease" in lower:
        return "Equity"
    if "ownership" in lower:
        return "Equity"

    # Partnership
    if "partnership" in lower:
        return "Development partnership"
    if "framework" in lower and "ppa" not in lower:
        return "Development partnership"

    # PPA
    if "ppa" in lower:
        return "PPA"
    if "agreement" in lower:
        return "PPA"
    if "contract" in lower:
        return "PPA"
    if "green tariff" in lower:
        return "PPA"

    return None


def classify_technology(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if "solar" in text and "storage" in text:
        return "Storage"
    if "storage" in text or "battery" in text:
        return "Storage"
    if "offshore" in text:
        return "Wind"
    if "wind" in text:
        return "Wind"
    if "solar" in text:
        return "Solar"
    if "nuclear" in text or "smr" in text or "fusion" in text:
        return "Nuclear"
    if "geothermal" in text:
        return "Geothermal"

    return None


# ============================================================
# LOAD DATA
# ============================================================

def parse_google() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_FILE, sheet_name="Google")
    df = df.rename(columns=normalize_column_name)

    return pd.DataFrame({
        "company": "Google",
        "project": df["Project / Deal Name (as named in report)"],
        "year": df["Report Year"].map(extract_year),
        "instrument_raw": df["Financial instrument"],
        "technology_raw": df["Technology Type"],
    })


def parse_meta() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_FILE, sheet_name="Meta")
    df = df.rename(columns=normalize_column_name)

    return pd.DataFrame({
        "company": "Meta",
        "project": df["Project"],
        "year": df["Year (Operational/Announced)"].map(extract_year),
        "instrument_raw": df["Financial Instrument / Partner"],
        "technology_raw": df["Technology"],
    })


def parse_microsoft() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_FILE, sheet_name="Microsoft")
    df = df.rename(columns=normalize_column_name)

    return pd.DataFrame({
        "company": "Microsoft",
        "project": df["Project"],
        "year": df["Year (Operational/Announced)"].map(extract_year),
        "instrument_raw": df["Financial Instrument / Partner"],
        "technology_raw": df["Technology"],
    })


def parse_amazon() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_FILE, sheet_name="Amazon")
    df = df.rename(columns=normalize_column_name)

    return pd.DataFrame({
        "company": "Amazon",
        "project": df["Site Name"],
        "year": df["Operational Date"].map(extract_year),
        "instrument_raw": df["Financial Instrument"],
        "technology_raw": df["Project Type"],
    })


def build_dataset() -> pd.DataFrame:
    df = pd.concat(
        [parse_google(), parse_meta(), parse_microsoft(), parse_amazon()],
        ignore_index=True
    )

    df["instrument"] = df["instrument_raw"].map(classify_instrument)
    df["technology"] = df["technology_raw"].map(classify_technology)

    df = df[df["instrument"].notna()].copy()
    df = df[df["technology"].notna()].copy()

    return df


# ============================================================
# AGGREGATION
# ============================================================

def technology_instrument_shares(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["technology", "instrument"])
          .size()
          .unstack(fill_value=0)
          .reindex(index=TECH_ORDER, columns=INSTRUMENT_ORDER, fill_value=0)
    )

    shares = counts.div(counts.sum(axis=1).replace(0, 1), axis=0) * 100
    return shares


# ============================================================
# LABELS
# ============================================================

def add_bar_labels(ax: plt.Axes, shares: pd.DataFrame) -> None:
    lefts = pd.Series(0.0, index=shares.index)

    for instrument in INSTRUMENT_ORDER:
        vals = shares[instrument]

        for y, tech in enumerate(shares.index):
            value = vals.loc[tech]
            if value >= 8:
                ax.text(
                    lefts.loc[tech] + value / 2,
                    y,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold"
                )
        lefts += vals


# ============================================================
# PLOT
# ============================================================

def plot_chart(df: pd.DataFrame) -> None:
    shares = technology_instrument_shares(df)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    bg = "#FFFFFF"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    lefts = pd.Series(0.0, index=shares.index)

    for instrument in INSTRUMENT_ORDER:
        ax.barh(
            shares.index,
            shares[instrument].values,
            left=lefts.values,
            color=COLORS[instrument],
            edgecolor="#F5F5F5",
            linewidth=1.0,
            height=0.52,
            label=instrument
        )
        lefts += shares[instrument]

    add_bar_labels(ax, shares)

    ax.set_title(
        TITLE,
        fontsize=17,
        fontweight="bold",
        pad=14,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#E3E7EC", edgecolor="#B9C3CF")
    )
    ax.set_xlabel("Share by Technology (%)", fontsize=12, color="#444444")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="both", labelsize=11, colors="#444444")

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B7BEC8")
    ax.spines["bottom"].set_color("#B7BEC8")
    ax.grid(False)
    ax.invert_yaxis()  # Solar at top

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False,
        fontsize=11
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = build_dataset()
    plot_chart(df)
    print(f"Saved chart to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()