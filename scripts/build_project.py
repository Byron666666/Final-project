from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


RAW_DATA_CANDIDATES = [
    ROOT / "YRBS_2007 (1).csv",
    ROOT / "YRBS_2007.csv",
    ROOT / "data" / "YRBS_2007 (1).csv",
    ROOT / "data" / "YRBS_2007.csv",
]
RAW_DATA = next((path for path in RAW_DATA_CANDIDATES if path.exists()), RAW_DATA_CANDIDATES[0])
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
NOTEBOOKS_DIR = ROOT / "notebooks"

MODEL_FORMULA = (
    'bmi_pct ~ C(sleep_group, Treatment(reference="8+ hours")) '
    "+ physical_activity_days + tv_hours + computer_hours "
    '+ C(sex, Treatment(reference="Female")) '
    '+ C(grade, Treatment(reference="10th grade"))'
)

SLEEP_CODE = {
    1: ("4 or less hours", 4),
    2: ("5 hours", 5),
    3: ("6 hours", 6),
    4: ("7 hours", 7),
    5: ("8 hours", 8),
    6: ("9 hours", 9),
    7: ("10 or more hours", 10),
}
HOURS_CODE = {
    1: ("None", 0.0),
    2: ("Less than 1 hour", 0.5),
    3: ("1 hour", 1.0),
    4: ("2 hours", 2.0),
    5: ("3 hours", 3.0),
    6: ("4 hours", 4.0),
    7: ("5 or more hours", 5.0),
}
SEX_CODE = {1: "Female", 2: "Male"}
GRADE_CODE = {
    1: "9th grade",
    2: "10th grade",
    3: "11th grade",
    4: "12th grade",
    5: "Ungraded/other",
}


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    return float(np.average(values[mask], weights=weights[mask]))


def p_label(p_value: float) -> str:
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "BMIPCT",
        "Sleep",
        "PhysicalActivity5OrMoreDays",
        "TelevisionWatching",
        "ComputerUse",
        "WhatIsYourSex",
        "InWhatGradeAreYou",
        "weight",
    ]
    raw = pd.read_csv(RAW_DATA, usecols=usecols)

    clean = pd.DataFrame(
        {
            "bmi_pct": raw["BMIPCT"],
            "sleep_code": raw["Sleep"],
            "physical_activity_code": raw["PhysicalActivity5OrMoreDays"],
            "tv_code": raw["TelevisionWatching"],
            "computer_code": raw["ComputerUse"],
            "sex": raw["WhatIsYourSex"].map(SEX_CODE),
            "grade": raw["InWhatGradeAreYou"].map(GRADE_CODE),
            "survey_weight": raw["weight"],
        }
    )

    clean["sleep_label"] = clean["sleep_code"].map(
        {code: label for code, (label, _) in SLEEP_CODE.items()}
    )
    clean["sleep_hours"] = clean["sleep_code"].map(
        {code: hours for code, (_, hours) in SLEEP_CODE.items()}
    )
    clean["sleep_group"] = pd.Categorical(
        np.select(
            [
                clean["sleep_hours"].le(6),
                clean["sleep_hours"].eq(7),
                clean["sleep_hours"].ge(8),
            ],
            ["<=6 hours", "7 hours", "8+ hours"],
            default=np.nan,
        ),
        categories=["<=6 hours", "7 hours", "8+ hours"],
        ordered=True,
    )

    clean["physical_activity_days"] = clean["physical_activity_code"].map(
        {code: code - 1 for code in range(1, 9)}
    )
    clean["tv_label"] = clean["tv_code"].map(
        {code: label for code, (label, _) in HOURS_CODE.items()}
    )
    clean["tv_hours"] = clean["tv_code"].map(
        {code: hours for code, (_, hours) in HOURS_CODE.items()}
    )
    clean["computer_label"] = clean["computer_code"].map(
        {code: label for code, (label, _) in HOURS_CODE.items()}
    )
    clean["computer_hours"] = clean["computer_code"].map(
        {code: hours for code, (_, hours) in HOURS_CODE.items()}
    )
    clean["screen_time_hours"] = clean["tv_hours"] + clean["computer_hours"]
    clean["adequate_sleep"] = np.where(
        clean["sleep_hours"].ge(8), "8+ hours", "Less than 8 hours"
    )
    clean.loc[clean["sleep_hours"].isna(), "adequate_sleep"] = np.nan

    clean["grade"] = pd.Categorical(
        clean["grade"],
        categories=["9th grade", "10th grade", "11th grade", "12th grade"],
        ordered=True,
    )
    clean["sex"] = pd.Categorical(clean["sex"], categories=["Female", "Male"])

    processed_path = PROCESSED_DIR / "yrbs_2007_sleep_bmi_cleaned.csv"
    clean.to_csv(processed_path, index=False)

    model_columns = [
        "bmi_pct",
        "sleep_group",
        "physical_activity_days",
        "tv_hours",
        "computer_hours",
        "sex",
        "grade",
        "survey_weight",
    ]
    analysis = clean.dropna(subset=model_columns).copy()
    analysis.to_csv(PROCESSED_DIR / "yrbs_2007_sleep_bmi_analysis_sample.csv", index=False)
    return clean, analysis


def fit_models(analysis: pd.DataFrame) -> dict[str, object]:
    ols = smf.ols(MODEL_FORMULA, data=analysis).fit(cov_type="HC3")
    wls = smf.wls(MODEL_FORMULA, data=analysis, weights=analysis["survey_weight"]).fit(
        cov_type="HC3"
    )
    anova_model = smf.ols("bmi_pct ~ C(sleep_group)", data=analysis).fit()
    anova = anova_lm(anova_model, typ=2)
    groups = [g["bmi_pct"].values for _, g in analysis.groupby("sleep_group", observed=True)]
    f_stat, p_value = stats.f_oneway(*groups)
    return {
        "ols": ols,
        "wls": wls,
        "anova": anova,
        "anova_f": float(f_stat),
        "anova_p": float(p_value),
    }


def coefficient_table(model, model_name: str) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": model.params.values,
            "robust_se": model.bse.values,
            "p_value": model.pvalues.values,
            "ci_low": model.conf_int()[0].values,
            "ci_high": model.conf_int()[1].values,
            "model": model_name,
        }
    )
    return table


def build_tables(
    clean: pd.DataFrame, analysis: pd.DataFrame, models: dict[str, object]
) -> dict[str, pd.DataFrame]:
    variable_dictionary = pd.DataFrame(
        [
            {
                "analysis_variable": "bmi_pct",
                "source_column": "BMIPCT",
                "definition": "BMI percentile based on CDC age-sex growth charts.",
            },
            {
                "analysis_variable": "sleep_group",
                "source_column": "Sleep / Q97",
                "definition": "<=6 hours, 7 hours, or 8+ hours on an average school night.",
            },
            {
                "analysis_variable": "physical_activity_days",
                "source_column": "PhysicalActivity5OrMoreDays / Q80",
                "definition": "Days in the past 7 days with at least 60 minutes of physical activity.",
            },
            {
                "analysis_variable": "tv_hours",
                "source_column": "TelevisionWatching / Q81",
                "definition": "Approximate TV hours on an average school day; 5+ coded as 5.",
            },
            {
                "analysis_variable": "computer_hours",
                "source_column": "ComputerUse / Q82",
                "definition": "Approximate non-school computer/video game hours; 5+ coded as 5.",
            },
            {
                "analysis_variable": "sex",
                "source_column": "WhatIsYourSex / Q2",
                "definition": "1 = Female, 2 = Male.",
            },
            {
                "analysis_variable": "grade",
                "source_column": "InWhatGradeAreYou / Q3",
                "definition": "9th through 12th grade; ungraded/other excluded from regression.",
            },
        ]
    )

    summary = pd.DataFrame(
        [
            {"metric": "Original records", "value": len(clean)},
            {"metric": "Records with BMI percentile", "value": clean["bmi_pct"].notna().sum()},
            {"metric": "Complete regression records", "value": len(analysis)},
            {"metric": "Mean BMI percentile, analysis sample", "value": analysis["bmi_pct"].mean()},
            {
                "metric": "Weighted mean BMI percentile, analysis sample",
                "value": weighted_mean(analysis["bmi_pct"], analysis["survey_weight"]),
            },
            {
                "metric": "Short sleep share (<=6 hours), analysis sample",
                "value": (analysis["sleep_group"].eq("<=6 hours")).mean(),
            },
        ]
    )

    sleep_summary = (
        analysis.groupby("sleep_group", observed=True)
        [["bmi_pct", "survey_weight", "screen_time_hours", "physical_activity_days"]]
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "percent": 100 * len(g) / len(analysis),
                    "bmi_mean": g["bmi_pct"].mean(),
                    "bmi_sd": g["bmi_pct"].std(),
                    "bmi_median": g["bmi_pct"].median(),
                    "weighted_bmi_mean": weighted_mean(g["bmi_pct"], g["survey_weight"]),
                    "mean_screen_time_hours": g["screen_time_hours"].mean(),
                    "mean_physical_activity_days": g["physical_activity_days"].mean(),
                }
            )
        )
        .reset_index()
    )

    screen_summary = (
        analysis.assign(
            screen_time_bin=pd.cut(
                analysis["screen_time_hours"],
                bins=[-0.01, 1.99, 3.99, 5.99, 10.01],
                labels=["0-<2", "2-<4", "4-<6", "6+"],
            )
        )
        .groupby("screen_time_bin", observed=True)
        [["bmi_pct", "survey_weight"]]
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "bmi_mean": g["bmi_pct"].mean(),
                    "weighted_bmi_mean": weighted_mean(g["bmi_pct"], g["survey_weight"]),
                }
            )
        )
        .reset_index()
    )

    anova = models["anova"].reset_index().rename(columns={"index": "term"})
    anova["p_value_scipy_f_oneway"] = models["anova_p"]

    coeffs = pd.concat(
        [
            coefficient_table(models["ols"], "OLS robust HC3"),
            coefficient_table(models["wls"], "WLS survey weight robust HC3"),
        ],
        ignore_index=True,
    )

    model_metrics = pd.DataFrame(
        [
            {
                "model": "OLS robust HC3",
                "nobs": int(models["ols"].nobs),
                "r_squared": models["ols"].rsquared,
                "adj_r_squared": models["ols"].rsquared_adj,
            },
            {
                "model": "WLS survey weight robust HC3",
                "nobs": int(models["wls"].nobs),
                "r_squared": models["wls"].rsquared,
                "adj_r_squared": models["wls"].rsquared_adj,
            },
        ]
    )

    tables = {
        "variable_dictionary": variable_dictionary,
        "sample_summary": summary,
        "sleep_group_summary": sleep_summary,
        "screen_time_summary": screen_summary,
        "anova_sleep_group": anova,
        "model_coefficients": coeffs,
        "model_metrics": model_metrics,
    }
    for name, table in tables.items():
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    return tables


def create_figures(
    analysis: pd.DataFrame, tables: dict[str, pd.DataFrame], models: dict[str, object]
) -> dict[str, Path]:
    sns.set_theme(style="whitegrid", context="talk")
    palette = ["#3B82F6", "#38BDF8", "#14B8A6"]
    figure_paths: dict[str, Path] = {}

    sleep = tables["sleep_group_summary"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.bar(sleep["sleep_group"], sleep["weighted_bmi_mean"], color=palette, width=0.7)
    ax.set_ylim(55, 68)
    ax.set_ylabel("Weighted mean BMI percentile")
    ax.set_xlabel("Sleep on an average school night")
    ax.set_title("BMI Percentile by Sleep Group")
    for idx, row in sleep.iterrows():
        ax.text(
            idx,
            row["weighted_bmi_mean"] + 0.35,
            f"{row['weighted_bmi_mean']:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    fig.tight_layout()
    path = FIGURES_DIR / "bmi_by_sleep_group.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_paths["bmi_by_sleep_group"] = path

    screen = tables["screen_time_summary"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(
        screen["screen_time_bin"].astype(str),
        screen["weighted_bmi_mean"],
        marker="o",
        linewidth=3,
        color="#2563EB",
    )
    ax.set_ylabel("Weighted mean BMI percentile")
    ax.set_xlabel("Combined TV + computer/video game hours")
    ax.set_title("BMI Percentile Increases with Screen Time")
    ax.set_ylim(58, 70)
    for idx, row in screen.iterrows():
        ax.text(
            idx,
            row["weighted_bmi_mean"] + 0.25,
            f"{row['weighted_bmi_mean']:.1f}",
            ha="center",
            fontsize=11,
        )
    fig.tight_layout()
    path = FIGURES_DIR / "bmi_by_screen_time.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_paths["bmi_by_screen_time"] = path

    wls_coeffs = tables["model_coefficients"].query(
        "model == 'WLS survey weight robust HC3' and term != 'Intercept'"
    )
    label_map = {
        'C(sleep_group, Treatment(reference="8+ hours"))[T.<=6 hours]': "Sleep <=6 vs 8+",
        'C(sleep_group, Treatment(reference="8+ hours"))[T.7 hours]': "Sleep 7 vs 8+",
        "physical_activity_days": "Physical activity days",
        "tv_hours": "TV hours",
        "computer_hours": "Computer/video hours",
        'C(sex, Treatment(reference="Female"))[T.Male]': "Male vs female",
        'C(grade, Treatment(reference="10th grade"))[T.9th grade]': "9th vs 10th",
        'C(grade, Treatment(reference="10th grade"))[T.11th grade]': "11th vs 10th",
        'C(grade, Treatment(reference="10th grade"))[T.12th grade]': "12th vs 10th",
    }
    wls_coeffs = wls_coeffs.assign(label=wls_coeffs["term"].map(label_map))
    order = [
        "Sleep <=6 vs 8+",
        "Sleep 7 vs 8+",
        "TV hours",
        "Computer/video hours",
        "Physical activity days",
        "Male vs female",
        "9th vs 10th",
        "11th vs 10th",
        "12th vs 10th",
    ]
    wls_coeffs["label"] = pd.Categorical(wls_coeffs["label"], categories=order, ordered=True)
    wls_coeffs = wls_coeffs.sort_values("label")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    y = np.arange(len(wls_coeffs))
    ax.errorbar(
        wls_coeffs["estimate"],
        y,
        xerr=[
            wls_coeffs["estimate"] - wls_coeffs["ci_low"],
            wls_coeffs["ci_high"] - wls_coeffs["estimate"],
        ],
        fmt="o",
        color="#0F766E",
        ecolor="#99F6E4",
        elinewidth=4,
        capsize=3,
    )
    ax.axvline(0, color="#374151", linewidth=1)
    ax.set_yticks(y, wls_coeffs["label"])
    ax.set_xlabel("Change in BMI percentile")
    ax.set_title("Weighted Regression Coefficients")
    fig.tight_layout()
    path = FIGURES_DIR / "weighted_regression_coefficients.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_paths["weighted_regression_coefficients"] = path

    return figure_paths


def create_infographic(
    tables: dict[str, pd.DataFrame], models: dict[str, object], figure_paths: dict[str, Path]
) -> dict[str, Path]:
    coeffs = tables["model_coefficients"].query("model == 'WLS survey weight robust HC3'")
    short_term = 'C(sleep_group, Treatment(reference="8+ hours"))[T.<=6 hours]'
    tv_term = "tv_hours"
    short = coeffs.set_index("term").loc[short_term]
    tv = coeffs.set_index("term").loc[tv_term]
    metrics = tables["model_metrics"].query("model == 'WLS survey weight robust HC3'").iloc[0]
    sleep = tables["sleep_group_summary"]
    screen = tables["screen_time_summary"]

    fig = plt.figure(figsize=(8.5, 11), dpi=180, facecolor="#F8FAFC")
    fig.text(
        0.07,
        0.955,
        "Sleep, Screen Time, and BMI Percentile",
        fontsize=23,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.07,
        0.925,
        "YRBS 2007 National High School Survey | ANOVA + weighted regression",
        fontsize=11.5,
        color="#475569",
    )

    cards = [
        ("Analysis Sample", f"{int(metrics['nobs']):,}", "complete student records"),
        ("Short sleep", f"+{short['estimate']:.2f}", "BMI percentile vs 8+ hours"),
        ("TV time", f"+{tv['estimate']:.2f}", "BMI percentile per hour"),
    ]
    card_positions = [(0.07, 0.785, 0.26, 0.105), (0.37, 0.785, 0.26, 0.105), (0.67, 0.785, 0.26, 0.105)]
    for idx, (title, value, subtitle) in enumerate(cards):
        ax = fig.add_axes(card_positions[idx])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.06, 0.76, title, fontsize=10.5, color="#475569", transform=ax.transAxes)
        ax.text(
            0.06,
            0.39,
            value,
            fontsize=25,
            fontweight="bold",
            color="#0369A1",
            transform=ax.transAxes,
        )
        ax.text(0.06, 0.14, subtitle, fontsize=9.2, color="#64748B", transform=ax.transAxes)

    ax_bar = fig.add_axes((0.08, 0.475, 0.40, 0.255), facecolor="white")
    ax_bar.bar(
        sleep["sleep_group"],
        sleep["weighted_bmi_mean"],
        color=["#3B82F6", "#38BDF8", "#14B8A6"],
        width=0.65,
    )
    ax_bar.set_ylim(55, 68)
    ax_bar.set_ylabel("Weighted mean BMI percentile", fontsize=8.8)
    ax_bar.set_title("BMI by Sleep Group", fontsize=12, fontweight="bold", pad=8)
    ax_bar.tick_params(axis="x", labelsize=8.8)
    ax_bar.tick_params(axis="y", labelsize=8.8)
    ax_bar.grid(True, axis="y", color="#E2E8F0")
    ax_bar.grid(False, axis="x")
    for idx, row in sleep.iterrows():
        ax_bar.text(
            idx,
            row["weighted_bmi_mean"] + 0.35,
            f"{row['weighted_bmi_mean']:.1f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#334155",
        )

    ax_screen = fig.add_axes((0.56, 0.475, 0.36, 0.255), facecolor="white")
    ax_screen.plot(
        screen["screen_time_bin"].astype(str),
        screen["weighted_bmi_mean"],
        marker="o",
        linewidth=3,
        markersize=7,
        color="#2563EB",
    )
    ax_screen.set_ylim(58, 70)
    ax_screen.set_title("BMI by Screen Time", fontsize=12, fontweight="bold", pad=8)
    ax_screen.set_ylabel("Weighted mean BMI percentile", fontsize=8.8)
    ax_screen.set_xlabel("TV + computer/video hours", fontsize=8.8)
    ax_screen.tick_params(axis="both", labelsize=8.8)
    ax_screen.grid(True, color="#E2E8F0")
    for idx, row in screen.iterrows():
        ax_screen.text(
            idx,
            row["weighted_bmi_mean"] + 0.3,
            f"{row['weighted_bmi_mean']:.1f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#334155",
        )

    ax_question = fig.add_axes((0.07, 0.265, 0.40, 0.145))
    ax_question.axis("off")
    ax_question.text(
        0,
        0.95,
        "Research Question",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        transform=ax_question.transAxes,
    )
    ax_question.text(
        0,
        0.66,
        "Are sleep duration and lifestyle behaviors\nassociated with BMI percentile?",
        fontsize=10.5,
        color="#334155",
        linespacing=1.25,
        transform=ax_question.transAxes,
    )
    ax_question.text(
        0,
        0.24,
        "Outcome: BMI percentile\nPredictors: sleep, TV, computer/video use,\nphysical activity, sex, and grade",
        fontsize=9.5,
        color="#475569",
        linespacing=1.25,
        transform=ax_question.transAxes,
    )

    ax_results = fig.add_axes((0.55, 0.265, 0.38, 0.145))
    ax_results.axis("off")
    ax_results.text(
        0,
        0.95,
        "Adjusted Regression Results",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        transform=ax_results.transAxes,
    )
    result_lines = [
        (f"+{short['estimate']:.2f}", "Short sleep vs 8+ hours", p_label(float(short["p_value"]))),
        (f"+{tv['estimate']:.2f}", "Each additional TV hour", p_label(float(tv["p_value"]))),
        (f"{metrics['r_squared']:.3f}", "Model R-squared", ""),
    ]
    for idx, (value, label, pval) in enumerate(result_lines):
        y_pos = 0.66 - idx * 0.25
        ax_results.text(
            0,
            y_pos,
            value,
            fontsize=15,
            fontweight="bold",
            color="#0369A1",
            transform=ax_results.transAxes,
        )
        ax_results.text(
            0.22,
            y_pos + 0.02,
            label if not pval else f"{label} | p={pval}",
            fontsize=9.8,
            color="#334155",
            transform=ax_results.transAxes,
        )

    ax_takeaway = fig.add_axes((0.07, 0.115, 0.86, 0.095))
    ax_takeaway.set_facecolor("#E0F2FE")
    for spine in ax_takeaway.spines.values():
        spine.set_visible(False)
    ax_takeaway.set_xticks([])
    ax_takeaway.set_yticks([])
    ax_takeaway.text(
        0.03,
        0.66,
        "Takeaway",
        fontsize=12,
        fontweight="bold",
        color="#0F172A",
        transform=ax_takeaway.transAxes,
    )
    ax_takeaway.text(
        0.03,
        0.25,
        "Shorter sleep and higher TV time are associated with higher BMI percentile.\n"
        "This is an association in a cross-sectional survey, not proof of causation.",
        fontsize=10.0,
        color="#334155",
        linespacing=1.25,
        transform=ax_takeaway.transAxes,
    )

    fig.text(
        0.07,
        0.065,
        f"ANOVA p-value for sleep groups: {p_label(models['anova_p'])}. "
        "Weighted regression uses survey weights and robust HC3 standard errors.",
        fontsize=9,
        color="#64748B",
    )

    png_path = ROOT / "one_page_summary.png"
    pdf_path = ROOT / "one_page_summary.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def create_notebook() -> Path:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# YRBS 2007 Final Project: Sleep, Screen Time, and BMI Percentile\n\n"
            "This notebook reproduces the final project workflow: research question, data cleaning, "
            "ANOVA, weighted regression, figures, and interpretation."
        ),
        nbf.v4.new_markdown_cell(
            "## Research Question\n\n"
            "Are sleep duration and lifestyle behaviors associated with BMI percentile among U.S. high school students "
            "in the 2007 National Youth Risk Behavior Survey?"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n\n"
            "ROOT = Path.cwd()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "sys.path.append(str(ROOT / 'scripts'))\n\n"
            "import build_project\n"
            "clean, analysis = build_project.load_and_clean()\n"
            "models = build_project.fit_models(analysis)\n"
            "tables = build_project.build_tables(clean, analysis, models)\n"
            "figure_paths = build_project.create_figures(analysis, tables, models)\n"
            "summary_paths = build_project.create_infographic(tables, models, figure_paths)\n"
            "analysis.shape"
        ),
        nbf.v4.new_markdown_cell("## Variables and Cleaning"),
        nbf.v4.new_code_cell("tables['variable_dictionary']"),
        nbf.v4.new_markdown_cell("## Descriptive Results"),
        nbf.v4.new_code_cell(
            "tables['sample_summary']\n\n"
            "tables['sleep_group_summary'].round(2)"
        ),
        nbf.v4.new_markdown_cell("## ANOVA"),
        nbf.v4.new_code_cell(
            "tables['anova_sleep_group'].round(4)\n"
            "print(f\"One-way ANOVA p-value: {models['anova_p']:.4f}\")"
        ),
        nbf.v4.new_markdown_cell("## Weighted Regression"),
        nbf.v4.new_code_cell(
            "coef = tables['model_coefficients']\n"
            "coef[coef['model'].eq('WLS survey weight robust HC3')].round(4)"
        ),
        nbf.v4.new_markdown_cell("## Figures"),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n\n"
            "display(Image(filename=str(figure_paths['bmi_by_sleep_group'])))\n"
            "display(Image(filename=str(figure_paths['bmi_by_screen_time'])))\n"
            "display(Image(filename=str(figure_paths['weighted_regression_coefficients'])))"
        ),
        nbf.v4.new_markdown_cell(
            "## Conclusion\n\n"
            "Students reporting <=6 hours of sleep had a higher BMI percentile than students reporting 8+ hours, "
            "after adjusting for TV time, computer/video game time, physical activity, sex, and grade. "
            "TV time was also positively associated with BMI percentile. These are associations in a cross-sectional "
            "survey, so they should not be interpreted as causal effects."
        ),
    ]
    path = NOTEBOOKS_DIR / "YRBS_2007_sleep_bmi_project.ipynb"
    nbf.write(nb, path)
    return path


def main() -> None:
    ensure_dirs()
    clean, analysis = load_and_clean()
    models = fit_models(analysis)
    tables = build_tables(clean, analysis, models)
    figure_paths = create_figures(analysis, tables, models)
    summary_paths = create_infographic(tables, models, figure_paths)
    notebook_path = create_notebook()

    coeffs = tables["model_coefficients"].query("model == 'WLS survey weight robust HC3'")
    short_term = 'C(sleep_group, Treatment(reference="8+ hours"))[T.<=6 hours]'
    tv_term = "tv_hours"
    short = coeffs.set_index("term").loc[short_term]
    tv = coeffs.set_index("term").loc[tv_term]
    print("Final project materials generated.")
    print(f"Analysis sample: {len(analysis):,} records")
    print(
        f"Short sleep coefficient: {short['estimate']:.2f} "
        f"(p={p_label(float(short['p_value']))})"
    )
    print(f"TV-hours coefficient: {tv['estimate']:.2f} (p={p_label(float(tv['p_value']))})")
    print(f"ANOVA p-value: {p_label(models['anova_p'])}")
    print(f"Notebook: {notebook_path.relative_to(ROOT)}")
    print(f"One-page summary: {summary_paths['png'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
