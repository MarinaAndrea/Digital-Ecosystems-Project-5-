#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


OUTPUT_DIR = Path(__file__).resolve().parent
SOURCE_WORKBOOK = OUTPUT_DIR / "dataset digiecos.xlsx"

PRE_PERIOD = "2020-2021"
POST_PERIOD = "2024-2026"
COMPANIES = ["Microsoft", "Meta", "Google", "Amazon"]

BROAD_BUCKET_ORDER = ["Solar", "Wind", "Hydro", "Storage", "Nuclear", "Geothermal", "Other"]

BROAD_BUCKET_COLORS = {
    "Solar": "#F9A21B",
    "Wind": "#4F8BCF",
    "Hydro": "#2AA3A1",
    "Storage": "#63C46B",
    "Nuclear": "#B84FB5",
    "Geothermal": "#E46A43",
    "Other": "#8F8F8F",
}

OTHER_DISPLAY_THRESHOLD = 5.0
SEGMENT_LABEL_THRESHOLD = 8.0

GOOGLE_LEGACY_MARKERS = [
    "signed 2010",
    "signed 2015",
    "signed ~2015",
    "signed ~2016",
    "signed ~2017",
    "signed ~2019",
    "pre-2017",
    "pre-existing ppa",
    "not a new 2023 deal",
    "not a new 2024 deal",
    "same ppa as",
    "photo caption",
    "photo details page only",
    "photo details appendix",
    "appendix section header photo",
    "first dutch ppa",
    "first ppa signed 2010",
    "part of the sep 2019 package",
    "announced in sep 2019",
]

GOOGLE_REPORT_FALLBACK_MARKERS = [
    "missing from the verification sheet",
    "new information",
    "first polish ppa",
    "first irish solar ppa",
    "first google contract in uk",
    "first google contract in spain",
    "commercial-scale ctt agreement",
    "landmark advanced nuclear deal",
    "expected operational 2025",
]

GOOGLE_OPERATIONAL_PATTERNS = [
    r"expected operational\s+(20\d{2})",
    r"operational(?:\s+nov)?\s+(20\d{2})",
    r"became operational(?:\s+in)?\s+(20\d{2})",
    r"came online(?:\s+in)?\s+(20\d{2})",
    r"commissioned(?:\s+mid-)?\s*(20\d{2})",
    r"completed(?:\s+nov)?\s*(20\d{2})",
]

GOOGLE_SIGNED_PATTERNS = [
    r"\bsigned(?:\s+alongside|\s+in|\s+~)?\s*(20\d{2})",
    r"\bannounced(?:\s+in|\s+alongside|\s+~)?\s*(20\d{2})",
    r"\b(20\d{2})\b[^\n.]{0,40}\bannouncement\b",
    r"\b(20\d{2})\b[^\n.]{0,40}\bpackage\b",
    r"\(~?(20\d{2})\)",
]

GOOGLE_YEAR_OVERRIDES = {
    "Tainan City Solar, Taiwan — 10 MW": (2019.0, "signed/announced"),
    "TVA Alabama Solar (NextEra) — 150 MW": (2019.0, "signed/announced"),
    "TVA Tennessee Solar (Invenergy) — 150 MW": (2019.0, "signed/announced"),
    "BlackRock / New Green Power, Taiwan — 1 GW Solar": (2024.0, "report-year"),
}

GOOGLE_BASIS_PRIORITY = {
    "operational": 0,
    "signed/announced": 1,
    "report-year": 2,
    "legacy-repeat": 3,
    "unknown": 4,
}


def normalize_number_text(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return text.replace(",", ".")


def normalize_column_name(column_name: object) -> str:
    return str(column_name).strip().strip('"').strip("'")


def extract_year(value: object) -> float | None:
    if pd.isna(value):
        return None
    match = re.search(r"(20\d{2})", str(value))
    return float(match.group(1)) if match else None


def parse_first_number(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = normalize_number_text(str(value)).strip().lower()
    if text in {"", "nan", "n.a."}:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def parse_capacity_mw(value: object) -> float | None:
    if pd.isna(value):
        return None

    text = normalize_number_text(str(value)).strip().lower()
    if not text:
        return None
    if any(marker in text for marker in ["not specified", "gallons", "kwh/year"]):
        return None
    if "mw" not in text and "gw" not in text:
        return None

    text_outside_parentheses = re.sub(r"\([^)]*\)", "", text)
    multiple_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(gw|mw)", text_outside_parentheses)
    if "+" in text_outside_parentheses and len(multiple_matches) >= 2:
        total_mw = 0.0
        for number_text, unit in multiple_matches:
            number = float(number_text)
            total_mw += number * 1000 if unit == "gw" else number
        return total_mw

    first_match = re.search(r"(\d+(?:\.\d+)?)\s*(gw|mw)", text)
    if not first_match:
        return None

    number = float(first_match.group(1))
    unit = first_match.group(2)
    return number * 1000 if unit == "gw" else number


def normalize_project_type(value: object, *, amazon: bool = False) -> str | None:
    text = "" if pd.isna(value) else str(value).strip()
    lower_text = text.lower()

    if amazon and "on-site solar" in lower_text:
        return "On-site Solar"
    if "solar" in lower_text and "storage" in lower_text:
        return "Solar + Storage"
    if "offshore" in lower_text:
        return "Offshore Wind"
    if "wind" in lower_text and "solar" in lower_text:
        return "Wind + Solar"
    if "wind" in lower_text:
        return "Wind"
    if "solar" in lower_text:
        return "Solar"
    if "nuclear" in lower_text or "smr" in lower_text or "fusion" in lower_text:
        return "Nuclear"
    if "geothermal" in lower_text:
        return "Geothermal"
    if "hydro" in lower_text:
        return "Hydro"
    if "battery" in lower_text or "storage" in lower_text:
        return "Storage"
    if "biomass" in lower_text:
        return None
    return "Other"


def normalize_amazon_project_type(project_name: object, raw_type: object) -> str | None:
    base_type = normalize_project_type(raw_type, amazon=True)
    project_name_text = "" if pd.isna(project_name) else str(project_name).strip().lower()

    if "hydroelectric" in project_name_text:
        return "Hydro"
    if "offshore" in project_name_text:
        return "Offshore Wind"
    if "solar farm" in project_name_text and base_type in {"Wind", "Other", None}:
        return "Solar"
    if "wind farm" in project_name_text and base_type in {"Other", None}:
        return "Wind"
    return base_type


def normalize_sheet_project_type(project: object, technology: object, capacity: object) -> str | None:
    base_type = normalize_project_type(technology)
    combined_text = " ".join(
        str(value).lower() for value in (project, technology, capacity) if not pd.isna(value)
    )

    if "solar" in combined_text and ("battery" in combined_text or "storage" in combined_text):
        return "Solar + Storage"
    return base_type


def extract_google_event_year(project: object, report: object, notes: object, report_year: object) -> tuple[float | None, str]:
    project_name = "" if pd.isna(project) else str(project).strip()
    override = GOOGLE_YEAR_OVERRIDES.get(project_name)
    if override is not None:
        return override

    combined_text = f"{project} || {report} || {notes}".lower()

    for pattern in GOOGLE_OPERATIONAL_PATTERNS:
        match = re.search(pattern, combined_text)
        if match:
            return float(match.group(1)), "operational"

    for pattern in GOOGLE_SIGNED_PATTERNS:
        match = re.search(pattern, combined_text)
        if match:
            return float(match.group(1)), "signed/announced"

    highlight_match = re.search(r"\b(20\d{2})\s+highlights\b", combined_text)
    if highlight_match:
        return float(highlight_match.group(1)), "report-year"

    same_report_match = re.search(r"same ppa as the (20\d{2}) report entry", combined_text)
    if same_report_match:
        return float(same_report_match.group(1)), "report-year"

    if any(marker in combined_text for marker in GOOGLE_LEGACY_MARKERS):
        return None, "legacy-repeat"

    report_year_value = extract_year(report_year)
    if report_year_value is not None:
        explicit_years = [int(year) for year in re.findall(r"(20\d{2})", combined_text)]
        if any(marker in combined_text for marker in GOOGLE_REPORT_FALLBACK_MARKERS):
            return report_year_value, "report-year"
        if explicit_years and min(explicit_years) < int(report_year_value):
            return None, "unknown"
        if "photo" not in combined_text and "pre-existing" not in combined_text:
            return report_year_value, "report-year"

    return None, "unknown"


def canonicalize_google_project(project: object, capacity_mw: float | None) -> str:
    if pd.isna(project):
        return ""

    project_text = str(project).lower()
    project_text = project_text.replace("ø", "o").replace("ð", "d")

    alias_patterns = [
        (r"aes chile hybrid", "aes chile hybrid"),
        (r"piiparinm", "piiparinmaki wind farm"),
        (r"r[oø]dby fjord", "rodby fjord solar"),
        (r"golden hills", "golden hills wind farm"),
        (r"norther offshore", "norther offshore wind farm"),
        (r"el romero", "el romero solar farm"),
        (r"delfzijl", "delfzijl wind farm"),
        (r"kairos power", "kairos power smr"),
        (r"blackrock / new green power", "new green power taiwan"),
        (r"tullabeg", "tullabeg solar farm"),
        (r"przyr[oó]w", "przyrow wind farm"),
        (r"moray west", "moray west offshore wind"),
        (r"helena wind farm", "helena wind farm"),
        (r"texas ppa", "texas unnamed ppa 150"),
        (r"australia solar", "australia unnamed solar 25"),
    ]
    for pattern, alias in alias_patterns:
        if re.search(pattern, project_text):
            return alias

    if "fervo" in project_text:
        return "fervo ctt 115" if capacity_mw and capacity_mw >= 100 else "fervo pilot"
    if "edpr na" in project_text and capacity_mw == 500:
        return "edpr na 500 mw"
    if "1.5 gw" in project_text and "pjm" in project_text:
        return "pjm solar framework 1.5 gw"

    root_name = project_text.split("—")[0]
    root_name = root_name.split("(")[0]
    root_name = root_name.split(",")[0]
    return re.sub(r"\s+", " ", root_name).strip()


def load_amazon_records() -> pd.DataFrame:
    source_df = pd.read_excel(SOURCE_WORKBOOK, sheet_name="Amazon")
    source_df = source_df.rename(columns=normalize_column_name)
    records = pd.DataFrame(
        {
            "company": "Amazon",
            "project": source_df["Site Name"],
            "project_type": source_df.apply(
                lambda row: normalize_amazon_project_type(row["Site Name"], row["Project Type"]),
                axis=1,
            ),
            "event_year": source_df["Operational Date"].map(extract_year),
            "size_mw": source_df["System Size (MW)"].map(parse_first_number),
        }
    )

    has_tamega_hydro_row = records["project"].astype(str).str.contains(
        "Tamega Hydroelectric Complex",
        case=False,
        na=False,
    ).any()
    if has_tamega_hydro_row:
        records = records[
            ~records["project"].astype(str).str.fullmatch(
                r"Amazon Wind Farm Portugal - Tamega",
                case=False,
            )
        ].copy()

    return records[["company", "project_type", "event_year", "size_mw"]]


def load_google_records() -> pd.DataFrame:
    source_df = pd.read_excel(SOURCE_WORKBOOK, sheet_name="Google")
    source_df = source_df.rename(columns=normalize_column_name)
    source_df = source_df[source_df["Financial instrument"].notna()].copy()
    source_df = source_df[~source_df["Financial instrument"].str.contains("Aggregated Stat", na=False)].copy()
    if "Notes" not in source_df.columns:
        source_df["Notes"] = ""

    source_df["size_mw"] = source_df["Capacity (MW Google)"].map(parse_first_number)
    event_rows = source_df.apply(
        lambda row: extract_google_event_year(
            row["Project / Deal Name (as named in report)"],
            row["Report"],
            row["Notes"],
            row["Report Year"],
        ),
        axis=1,
    )
    source_df[["event_year", "basis"]] = pd.DataFrame(event_rows.tolist(), index=source_df.index)
    source_df["canonical"] = source_df.apply(
        lambda row: canonicalize_google_project(
            row["Project / Deal Name (as named in report)"],
            row["size_mw"],
        ),
        axis=1,
    )
    source_df["project_type"] = source_df["Technology Type"].map(normalize_project_type)
    source_df["company"] = "Google"
    source_df["priority"] = source_df["basis"].map(GOOGLE_BASIS_PRIORITY).fillna(9)
    source_df = source_df[source_df["size_mw"].notna() & source_df["event_year"].notna()].copy()
    source_df = source_df.sort_values(["canonical", "priority", "event_year"])
    source_df = source_df.drop_duplicates("canonical", keep="first")

    return source_df[["company", "project_type", "event_year", "size_mw"]]


def load_company_records(sheet_name: str, company: str) -> pd.DataFrame:
    source_df = pd.read_excel(SOURCE_WORKBOOK, sheet_name=sheet_name)
    source_df = source_df.rename(columns=normalize_column_name)
    records = pd.DataFrame(
        {
            "company": company,
            "project": source_df["Project"],
            "project_type": source_df.apply(
                lambda row: normalize_sheet_project_type(row["Project"], row["Technology"], row["Capacity"]),
                axis=1,
            ),
            "event_year": source_df["Year (Operational/Announced)"].map(extract_year),
            "size_mw": source_df["Capacity"].map(parse_capacity_mw),
            "capacity_text": source_df["Capacity"].astype(str),
        }
    )

    project_names = records["project"].astype(str)
    capacity_text = records["capacity_text"].astype(str)
    project_drop_pattern = r"aggregate|portfolio|global|arrangements|study|program|framework|supply agreement|additions"
    capacity_drop_pattern = r"cumulative|projects|pipeline|supply chain|total portfolio|new deals|operating|studied|target"
    if company == "Microsoft":
        project_drop_pattern = r"global|framework|supply agreement|additions"

    is_aggregate = project_names.str.contains(project_drop_pattern, case=False, na=False) | capacity_text.str.contains(
        capacity_drop_pattern,
        case=False,
        na=False,
    )
    records = records[~is_aggregate].copy()
    return records[["company", "project_type", "event_year", "size_mw"]]


def build_clean_records() -> pd.DataFrame:
    all_records = pd.concat(
        [
            load_amazon_records(),
            load_google_records(),
            load_company_records("Microsoft", "Microsoft"),
            load_company_records("Meta", "Meta"),
        ],
        ignore_index=True,
    )
    all_records = all_records[all_records["event_year"].notna() & all_records["size_mw"].notna()].copy()
    all_records["event_year"] = all_records["event_year"].astype(int)
    all_records = all_records[
        (all_records["event_year"] >= 2020)
        & (all_records["event_year"] <= 2026)
        & (all_records["event_year"] != 2023)
    ].copy()
    all_records["era"] = all_records["event_year"].apply(
        lambda year: PRE_PERIOD if year <= 2021 else POST_PERIOD if year >= 2024 else None
    )
    return all_records[all_records["era"].notna()].copy()


def expand_broad_buckets(clean_records: pd.DataFrame) -> pd.DataFrame:
    expanded_rows: list[dict[str, object]] = []

    for row in clean_records.itertuples(index=False):
        size_mw = float(row.size_mw)
        allocations: list[tuple[str, float]]

        if row.project_type in {"Solar", "On-site Solar"}:
            allocations = [("Solar", size_mw)]
        elif row.project_type in {"Wind", "Offshore Wind"}:
            allocations = [("Wind", size_mw)]
        elif row.project_type == "Hydro":
            allocations = [("Hydro", size_mw)]
        elif row.project_type == "Solar + Storage":
            allocations = [("Solar", size_mw / 2), ("Storage", size_mw / 2)]
        elif row.project_type == "Wind + Solar":
            allocations = [("Wind", size_mw / 2), ("Solar", size_mw / 2)]
        elif row.project_type == "Storage":
            allocations = [("Storage", size_mw)]
        elif row.project_type == "Nuclear":
            allocations = [("Nuclear", size_mw)]
        elif row.project_type == "Geothermal":
            allocations = [("Geothermal", size_mw)]
        elif row.project_type == "Other":
            allocations = [("Other", size_mw)]
        else:
            continue

        for bucket, allocated_mw in allocations:
            expanded_rows.append(
                {
                    "company": row.company,
                    "era": row.era,
                    "bucket": bucket,
                    "size_mw": allocated_mw,
                }
            )

    return pd.DataFrame(expanded_rows)


def summarize_bucket_shares(clean_records: pd.DataFrame) -> pd.DataFrame:
    expanded = expand_broad_buckets(clean_records)
    totals = (
        expanded.groupby(["company", "era", "bucket"], as_index=False)
        .agg(total_mw=("size_mw", "sum"))
    )
    company_period_totals = totals.groupby(["company", "era"])["total_mw"].transform("sum")
    totals["share_pct"] = totals["total_mw"] / company_period_totals * 100
    return totals


def select_plot_buckets(bucket_shares: pd.DataFrame) -> list[str]:
    plot_buckets = list(BROAD_BUCKET_ORDER)
    other_rows = bucket_shares[bucket_shares["bucket"] == "Other"]
    if other_rows.empty or other_rows["share_pct"].max() < OTHER_DISPLAY_THRESHOLD:
        plot_buckets.remove("Other")
    return plot_buckets


def build_plot_shares(bucket_shares: pd.DataFrame, plot_buckets: list[str]) -> pd.DataFrame:
    plot_shares = bucket_shares[bucket_shares["bucket"].isin(plot_buckets)].copy()
    visible_totals = plot_shares.groupby(["company", "era"])["share_pct"].transform("sum")
    plot_shares["share_pct"] = plot_shares["share_pct"] / visible_totals * 100
    return plot_shares


def plot_share_stacks(
    axis: plt.Axes,
    plot_shares: pd.DataFrame,
    plot_buckets: list[str],
    era: str,
    panel_title: str,
) -> None:
    era_totals = plot_shares[plot_shares["era"] == era].copy()
    pivot = (
        era_totals.pivot(index="company", columns="bucket", values="share_pct")
        .reindex(index=COMPANIES, columns=plot_buckets, fill_value=0)
        .fillna(0.0)
    )

    y_positions = list(range(len(COMPANIES)))
    left_offsets = pd.Series(0.0, index=COMPANIES)
    for bucket in plot_buckets:
        values = pivot[bucket]
        if values.sum() == 0:
            continue
        axis.barh(
            y_positions,
            values.values,
            left=left_offsets.values,
            color=BROAD_BUCKET_COLORS[bucket],
            edgecolor="white",
            linewidth=0.8,
            height=0.5,
            label=bucket,
        )
        centers = left_offsets + values / 2
        for y_position, center, share in zip(y_positions, centers, values):
            if share >= SEGMENT_LABEL_THRESHOLD:
                axis.text(center, y_position, f"{int(round(share))}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left_offsets += values

    axis.set_yticks(y_positions)
    axis.set_yticklabels(COMPANIES, fontweight="bold")
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(axis="x", color="#E1E6F0", linewidth=1.0, linestyle="--")
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#C7CDD8")
    axis.spines["bottom"].set_color("#C7CDD8")
    axis.set_xlabel("Share of Clean Energy Mix (%)")
    axis.set_title(
        panel_title,
        fontsize=13,
        fontweight="bold",
        pad=10,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F4F7FC", "edgecolor": "#C9D3E6"},
    )


def build_company_combined_chart(
    bucket_shares: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_buckets = select_plot_buckets(bucket_shares)
    plot_shares = build_plot_shares(bucket_shares, plot_buckets)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), sharey=False)

    plot_share_stacks(
        axes[0],
        plot_shares,
        plot_buckets,
        PRE_PERIOD,
        "Before AI Boom  (2020 - 2021)",
    )
    plot_share_stacks(
        axes[1],
        plot_shares,
        plot_buckets,
        POST_PERIOD,
        "After AI Boom  (2024 - 2026)",
    )
    axes[1].tick_params(axis="y", labelleft=True)

    legend_map: dict[str, object] = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        legend_map.update(dict(zip(labels, handles)))
    ordered_labels = [label for label in plot_buckets if label in legend_map]
    ordered_handles = [legend_map[label] for label in ordered_labels]
    fig.legend(
        ordered_handles,
        ordered_labels,
        ncol=len(ordered_labels),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.suptitle(
        "Hyperscaler Clean Energy Mix by Technology Type",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    clean_records = build_clean_records()
    bucket_shares = summarize_bucket_shares(clean_records)
    output_path = OUTPUT_DIR / "investment_distribution_by_company_combined.png"
    build_company_combined_chart(bucket_shares, output_path)
    print(f"Wrote {output_path.name}")


if __name__ == "__main__":
    main()
