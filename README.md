# YRBS 2007 Final Project: Sleep, Screen Time, and BMI Percentile

## Student Information

Name: TODO

Student ID: TODO

## Project Repository

GitHub repository: TODO

## Presentation Video

YouTube link: TODO

## Overview

This project analyzes the 2007 National Youth Risk Behavior Survey (YRBS) to study whether sleep duration and lifestyle behaviors are associated with BMI percentile among U.S. high school students.

Research question: Are sleep duration and lifestyle behaviors associated with BMI percentile?

## Variables

Outcome variable:

- `BMIPCT`: BMI percentile based on CDC age-sex growth charts.

Main predictors and controls:

- `Sleep` / Q97: average hours of sleep on a school night, recoded as `<=6 hours`, `7 hours`, and `8+ hours`.
- `TelevisionWatching` / Q81: approximate TV hours on an average school day.
- `ComputerUse` / Q82: approximate non-school computer/video game hours on an average school day.
- `PhysicalActivity5OrMoreDays` / Q80: days in the past 7 days with at least 60 minutes of physical activity.
- `WhatIsYourSex` / Q2 and `InWhatGradeAreYou` / Q3: demographic controls.

## Method

The analysis uses:

- One-way ANOVA to compare BMI percentile across sleep groups.
- Weighted least squares regression using the survey weight variable, with robust HC3 standard errors.

This is a cross-sectional observational analysis, so results should be interpreted as associations, not causal effects.

## Key Results

- Final regression sample: 11,256 complete student records.
- Weighted mean BMI percentile was 64.4 for students sleeping `<=6 hours`, 63.8 for `7 hours`, and 63.0 for `8+ hours`.
- ANOVA found a statistically significant difference in BMI percentile across sleep groups: p = 0.029.
- In the weighted regression, students sleeping `<=6 hours` had BMI percentile 2.36 points higher than students sleeping `8+ hours`, controlling for activity, screen time, sex, and grade: p = 0.005.
- Each additional TV hour was associated with a 2.16 point higher BMI percentile: p < 0.001.
- The model R-squared was 0.021, so these factors are statistically related to BMI percentile but explain only a small share of overall variation.

## Conclusion

Shorter sleep and higher TV time are associated with higher BMI percentile among students in this dataset. The result supports the idea that sleep and sedentary behavior are relevant to adolescent health, while also showing that BMI is influenced by many other factors not captured in this simple model.

## Repository Contents

- `notebooks/YRBS_2007_sleep_bmi_project.ipynb`: Jupyter notebook with the full workflow.
- `scripts/build_project.py`: Reproducible script that cleans data, runs models, makes tables, makes figures, and regenerates the notebook.
- `data/processed/`: cleaned data and complete-case analysis data.
- `tables/`: model coefficients, ANOVA table, sample summary, and variable dictionary.
- `figures/`: charts used in the project.
- `one_page_summary.png` and `one_page_summary.pdf`: one-page infographic summary.
- `presentation_outline.md`: optional English presentation outline for recording the required video.
- `Final Project - Instruction.pdf`: course project instructions.
- `YRBS_2007_National_User_Guide.pdf`: dataset user guide.
- `YRBS_2007 (1).csv`: original dataset.

## Reproduce the Analysis

Install the required Python packages, then run:

```bash
python scripts/build_project.py
```

On this computer, the Anaconda Python path also works:

```bash
/Users/e207309gmail.com/anaconda3/bin/python scripts/build_project.py
```

## Video Checklist

Before submission, record and upload the presentation video in English. The video should be at least 5 minutes, begin with full name and student ID, show the student's real face, use the student's real voice, explain the one-page summary, and show the key code used in the project.
