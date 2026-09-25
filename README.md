# Psychological Age in UK Biobank

*Analysis workflow · Prediction, interpretation, and health associations*

[psychological_age_analysis.R](psychological_age_analysis.R) presents the analysis sequence below. Model settings, statistical formulas, and covariates are retained from the supplied code. Visualization and figure exports are omitted.

## Workflow

| Module | Analysis | Main operations |
| :---: | :--- | :--- |
| **1** | Input checks and candidate learners | Check numeric inputs, remove constant features, map feature names, and define 20 regression learners. |
| **2** | Model selection and test evaluation | Use an 80/20 split; compare learners by five-fold training cross-validation and select the minimum-MAE model for held-out evaluation. |
| **3** | Full-cohort predictions | Generate ten-fold out-of-fold (OOF) predictions and fit the selected learner to all participants. |
| **4** | Feature interpretation | Calculate LightGBM feature importance, SHAP contributions, and feature-level summaries. |
| **5** | PAA and phenotypes | Derive psychological age acceleration (PAA); regress PAA on each phenotype, age, and covariates; apply Benjamini–Hochberg correction. |
| **6** | Disease and mortality | Fit Cox, negative-binomial, and Fine–Gray models, including two-year lag and baseline disease-burden analyses. |
| **7** | Diagnostics and nonlinearity | Assess proportional hazards, three-degree-of-freedom natural splines, and four-knot restricted cubic splines; retain numerical predictions. |

## Measures and adjustment

**PAA** is the residual from `pred ~ age`. Phenotype models use PAA as the outcome. Disease and mortality models use standardized `PAA_z` as the exposure.

Adjustment includes age, sex, ethnicity, Townsend deprivation index, qualifications, income, assessment-centre region, employment, and smoking. The corresponding covariate fields are `sex`, `ethnicity_5cat`, `town_index`, `qual_category`, `income`, `assessment_centre`, `employment`, and `smoking`; assessment centres are grouped into seven regions.

> **Interpretation block:** Module 4 uses LightGBM-specific importance and contribution methods. It expects the model object used for interpretation to be compatible with LightGBM.

## Inputs and use

| Input | Expected content |
| :--- | :--- |
| Main data object | `datall$q_final$mhq2016`: numeric age `label` and questionnaire features, with participant IDs in row names; `datall$age_data_2016`: assessment dates, completion status, and age. |
| Covariates and phenotypes | Participant-level covariates and phenotype measurements linked by participant ID. |
| Annotations | Feature dictionary and phenotype annotations. |
| Clinical events | Participant IDs, hospital ICD-10 codes and event dates, and death dates. |

1. Replace descriptive `<PLACEHOLDERS>` with local input, output, and directory locations; repeated placeholders identify the same resource.
2. Make the required data and software available, then run modules in order. Official UKB schema URLs remain in the script for annotation downloads.
3. Inspect the exported analysis results. Plotting is not part of this distribution.

<details>
<summary><strong>Software</strong></summary>

R packages include `data.table`, `mlr3`, `mlr3learners`, `mlr3extralearners`, `lightgbm`, `parallel`, `survival`, `MASS`, `cmprsk`, and `rms`, plus the backend packages required by the configured learners. Natural splines use `splines`. Worker settings are retained in the script.

</details>

---

Participant-level data and computed results are not included. The analyses were not rerun as part of these presentation edits.
