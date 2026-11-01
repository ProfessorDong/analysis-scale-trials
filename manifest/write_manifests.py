"""Write one YAML manifest per trial from the specifications below.

The specifications are the adjudication. Every allocation variable, table, visit
class and variable role was set by reading the deposited data dictionary, CRF,
protocol or README (converted to text under manifest/dict/) together with the
column inventories in manifest/pds_inventory.txt and manifest/dryad_inventory.txt.
Rules are in manifest/RULES.md and are referred to by number in `why` fields.

Variable specification for wide tables, one entry per column:
    name: (timing, role, label, endpoint, order)
timing   pre | pre_dose | post
role     primary | secondary | safety_lab | vital | tumour | pro | clinical | baseline
endpoint groups repeated measurements of one quantity so that rule 10 can pick
         the first and last post-dose occasion; order ranks them in time.

Long tables (one row per participant per parameter per visit) carry a `visits`
block instead: `pre` and `pre_dose` are lists of labels; `post` is either an
ordered list of labels or a regex whose numeric groups give the time order.
"""
from __future__ import annotations

import pathlib
import yaml

OUT = pathlib.Path(__file__).resolve().parent / "trials"
OUT.mkdir(exist_ok=True)

WEEK = r"^Week (\d+)$"
CYCLE_DAY = r"^Cycle (\d+), day (\d+)$"


def V(timing, role, label, endpoint=None, order=0):
    return dict(timing=timing, role=role, label=label, endpoint=endpoint, order=order)


# ----------------------------------------------------------------------------
# Shared Amgen baseline block (demo.sas7bdat)
# ----------------------------------------------------------------------------
def amgen_demo(extra=None):
    v = {
        "AGE": V("pre", "baseline", "Age at randomization (DDT)"),
        "B_WEIGHT": V("pre", "baseline", "Baseline weight (DDT, B_ prefix = baseline)"),
        "B_HEIGHT": V("pre", "baseline", "Baseline height"),
        "B_BSA": V("pre", "baseline", "Baseline body surface area"),
        "DIAGMONS": V("pre", "baseline", "Months since primary diagnosis, (randomization date - DIAGDT)/30.5"),
        "METMONS": V("pre", "baseline", "Months since metastatic diagnosis"),
    }
    if extra:
        v.update(extra)
    return v


TRIALS = []

# ============================================================================
# PROJECT DATA SPHERE
# ============================================================================

TRIALS.append(dict(
    id="Breast_Allianc_2002_200", source="PDS", nct="NCT00041119",
    title="CALGB 40101: adjuvant AC versus paclitaxel in early breast cancer",
    status="excluded", why="rule 8: the deposit carries only survival and disease-free survival months (survmos, dfsmos) and categorical fields; no continuous outcome and no continuous pre-randomization variable",
))

TRIALS.append(dict(
    id="Breast_Allianc_2006_216", source="PDS", nct="NCT00265759",
    title="ACOSOG Z1031: neoadjuvant exemestane vs letrozole vs anastrozole",
    design="multiarm", unit="individual", status="included",
    arms=dict(file="NCT00265759.csv", subject="mask_id", column="randf",
              evidence="Data dictionary: RANDF = Treatment arm (Arm 1) Exemestane, (Arm 2) Letrozole, (Arm 3) Anastrozole"),
    tables=[dict(file="NCT00265759.csv", kind="wide", subject="mask_id", variables={
        "pki67": V("pre", "baseline", "PKI67 = Baseline Ki67 (data dictionary)", "Ki67", 0),
        "ski67": V("post", "secondary", "SKI67 = Post-surgery Ki67 score (data dictionary); surgery follows 16 weeks of allocated aromatase inhibitor", "Ki67", 1),
        "PREALLRE": V("pre", "baseline", "Baseline Allred score (0-8; fails rule 9)"),
        "SXALLRED": V("post", "secondary", "Post-surgery Allred score (0-8; fails rule 9)", "Allred", 1),
    })],
    notes="Primary endpoint was clinical response at 16 weeks (protocol); Ki67 at surgery is a secondary biomarker endpoint.",
))

TRIALS.append(dict(
    id="Breast_Allianc_2008_158", source="PDS", nct="NCT00785291",
    title="CALGB 40502: nab-paclitaxel or ixabepilone vs paclitaxel, metastatic breast",
    status="excluded", why="rule 8: efficacy file carries only censored months (survmos, fumos, pfsmos, ttfmos); cycles file is exposure",
))

TRIALS.append(dict(
    id="Breast_Allianc_2009_162", source="PDS", nct="NCT00861705",
    title="CALGB 40603: neoadjuvant carboplatin and/or bevacizumab, 2x2 factorial, triple-negative breast cancer",
    design="factorial", unit="individual", status="included",
    arms=dict(file="treated2.csv", subject="MASK_ID", type="factorial",
              factors={"factora": "carboplatin (1) vs none (0)", "factorb": "bevacizumab (1) vs none (0)"},
              evidence="Data dictionary: FACTORA/FACTORB = randomized factors; INDRX = 4-arm index"),
    tables=[dict(file="treated2.csv", kind="wide", subject="MASK_ID", variables={
        "RCB": V("post", "secondary", "Residual cancer burden index at surgery after neoadjuvant therapy (data dictionary: continuous)", "RCB", 1),
        "nodeaxilpos": V("post", "secondary", "Number of positive axillary nodes at surgery", "nodeaxilpos", 1),
        "nodesentpos": V("post", "secondary", "Number of positive sentinel nodes at surgery", "nodesentpos", 1),
        "tsizepe": V("pre", "baseline", "Baseline tumour size by physical exam (cm)"),
        "tsizemamm": V("pre", "baseline", "Baseline tumour size by mammogram"),
        "tsizeultra": V("pre", "baseline", "Baseline tumour size by ultrasound"),
        "tsizemri": V("pre", "baseline", "Baseline tumour size by MRI"),
    }, exclude={"bev_ndoses": "rule 8 exposure", "pac_ndoses": "rule 8 exposure", "nodeaxilexam": "rule 8 process (nodes examined)", "nodesentexam": "rule 8 process", "her2fish": "rule 9 (HER2 ratio, few distinct values in a HER2-negative trial)"})],
    notes="Primary endpoint was binary pathologic complete response (Sikov 2015). 454 enrolled, 443 began treatment; deposited file holds the 443.",
))

TRIALS.append(dict(
    id="Colorec_Allianc_1994_201", source="PDS", nct="NCT00002575",
    title="COST (INT-0146): laparoscopic-assisted vs open colectomy for colon cancer",
    design="parallel", unit="individual", status="included",
    arms=dict(file="master.csv", subject="ANON_ID", column="arm",
              evidence="Data dictionary: arm = Analysis Arm, A = Open, B = LAP"),
    tables=[dict(file="master.csv", kind="wide", subject="ANON_ID", variables={
        "surgtime": V("post", "secondary", "Length of surgery (minutes); not censored", "surgtime", 1),
        "los": V("post", "secondary", "Length of hospital stay (days); recovery outcome, not censored in the deposit", "los", 1),
        "daysnarc": V("post", "secondary", "Length of narcotic use (days)", "daysnarc", 1),
    }, exclude={"converte": "rule 8 code (reason for conversion)", "dfs_time": "rule 8 censored", "fu_time": "rule 8 censored", "pg_time": "rule 8 censored"})],
    notes="Primary endpoint was time to recurrence (non-inferiority). The three retained variables are secondary recovery outcomes. All 871 randomized retained (rule 2).",
))

TRIALS.append(dict(
    id="Colorec_Allianc_2004_161", source="PDS", nct="NCT00079274",
    title="N0147: adjuvant FOLFOX with or without cetuximab, stage III colon cancer",
    design="multiarm", unit="individual", status="negative_control_only",
    arms=dict(file="characteristic.csv", subject="mask_id", column="ARM",
              evidence="DDT: ARM = experimental arm A-F (arms B, C, E, F closed early; all randomized)"),
    tables=[dict(file="characteristic.csv", kind="wide", subject="mask_id", variables={
        "bmi2": V("pre", "baseline", "DDT: BMI (registration)"),
    }, exclude={"NUMCYCLE": "rule 8 exposure", "endat_time": "rule 8 days on therapy", "ENDATRSN": "rule 8 code"})],
    notes="Adjuvant trial: every other numeric variable is a censored event time. Contributes negative controls only.",
))

TRIALS.append(dict(
    id="Colorec_Amgen_2004_310", source="PDS", nct="NCT00113763",
    title="Amgen 20020408: panitumumab plus best supportive care vs best supportive care, metastatic colorectal",
    design="parallel", unit="individual", status="included",
    arms=dict(file="adsl_pds2019.sas7bdat", subject="SUBJID", column="TRT",
              evidence="DDT_408: TRT = randomized treatment; ATRT = actual treatment"),
    tables=[
        dict(file="adlb_pds2019.sas7bdat", kind="long", subject="SUBJID", param="LBTEST", value="LBSTRESN", visit="VISIT", role="safety_lab",
             visits=dict(pre=["Screening"], pre_dose=["Week 1"], post=WEEK),
             exclude_params={"Carcinoembryonic Antigen": "tumour marker retained as secondary"},
             param_roles={"Carcinoembryonic Antigen": "secondary"}),
        dict(file="adls_pds2019.sas7bdat", kind="long", subject="SUBJID", param="LSCAT", value="LSSLD", visit="VISIT", role="tumour",
             visits=dict(pre=["Screening"], pre_dose=[], post=WEEK),
             exclude_params={"New lesion": "rule 8: exists only after progression", "Non-target lesion": "rule 9: non-target lesions are not measured"}),
        dict(file="adpm_pds2019.sas7bdat", kind="wide_visit", subject="SUBJID", visit="VISIT", role="vital",
             visits=dict(pre=["Screening"], pre_dose=["Week 1"], post=WEEK),
             variables={"WEIGHT": V("post", "vital", "Body weight by visit (adpm)", "WEIGHT")}),
        dict(file="adsl_pds2019.sas7bdat", kind="wide", subject="SUBJID", variables={
            "AGE": V("pre", "baseline", "Age"), "B_WEIGHT": V("pre", "baseline", "Baseline weight"), "DIAGMONS": V("pre", "baseline", "Months since diagnosis")},
             exclude={"DTHDYX": "rule 8 censored", "PFSDYCR": "rule 8 censored"}),
    ],
))

AMGEN_LAB_EXCL = {"PR24HU": "rule 13 (69 rows)", "PROT_U": "rule 13", "HCGBETC": "rule 13", "TSH": "rule 13", "CRCL_E": "derived creatinine clearance (rule 8 derivation)"}

TRIALS.append(dict(
    id="Colorec_Amgen_2005_262", source="PDS", nct="NCT00115765",
    title="Amgen 20040249 (PACCE): panitumumab plus bevacizumab and chemotherapy vs bevacizumab and chemotherapy",
    design="parallel", unit="individual", status="included",
    arms=dict(file="demo.sas7bdat", subject="SUBJID", column="TRT",
              evidence="DDT: TRT = randomized treatment (TRTCD 11/12); ATRT = actual treatment"),
    tables=[
        dict(file="lab.sas7bdat", kind="long", subject="SUBJID", param="LBTESTCD", value="LBSTRESN", visit="VISIT", role="safety_lab",
             visits=dict(pre=["Screening"], pre_dose=[], post=WEEK), exclude_params=AMGEN_LAB_EXCL, param_roles={"CEA": "secondary"}),
        dict(file="vitals_v.sas7bdat", kind="long", subject="SUBJID", param="VSTESTCD", value="VSSTRES", visit="VISIT", role="vital",
             visits=dict(pre=["Screening"], pre_dose=["Week 0"], post=WEEK), exclude_params={"ECOG": "rule 9 ordinal 0-4"}),
        dict(file="lesion.sas7bdat", kind="long", subject="SUBJID", param="LSCAT", value="LSSLD", visit="VISIT", role="tumour",
             visits=dict(pre=["Screening"], pre_dose=[], post=WEEK),
             exclude_params={"New lesion": "rule 8", "Non-target lesion": "rule 9"}),
        dict(file="demo.sas7bdat", kind="wide", subject="SUBJID", variables=amgen_demo(),
             exclude={"DURSURG": "pre-randomization history: months from diagnosis to surgery (DDT) but derived and coarse; rule 8", "FSTOXDS": "rule 8 dose", "FSTIRDS": "rule 8 dose", "TRTDUR": "rule 8 exposure", "B_LESNM": "rule 9 count of lesions"}),
    ],
))

TRIALS.append(dict(
    id="Colorec_Amgen_2006_263", source="PDS", nct="NCT00339183",
    title="Amgen 20050181: panitumumab plus FOLFIRI vs FOLFIRI, second-line metastatic colorectal",
    design="parallel", unit="individual", status="included",
    arms=dict(file="demo.sas7bdat", subject="SUBJID", column="TRT",
              evidence="DDT: TRT = randomized treatment (TRTCD 40/41); ATRT = actual"),
    tables=[
        dict(file="lab.sas7bdat", kind="long", subject="SUBJID", param="LBTESTCD", value="LBSTRESN", visit="VISIT", role="safety_lab",
             visits=dict(pre=["Screening"], pre_dose=["Cycle 1, day 1"], post=CYCLE_DAY), exclude_params=AMGEN_LAB_EXCL, param_roles={"CEA": "secondary"}),
        dict(file="vitals_v.sas7bdat", kind="long", subject="SUBJID", param="VSTESTCD", value="VSSTRES", visit="VISIT", role="vital",
             visits=dict(pre=["Screening"], pre_dose=["Cycle 1, day 1"], post=CYCLE_DAY), exclude_params={"ECOG": "rule 9"}),
        dict(file="lesion.sas7bdat", kind="long", subject="SUBJID", param="LSCAT", value="LSSLD", visit="VISIT", role="tumour",
             visits=dict(pre=["Screening"], pre_dose=[], post=WEEK),
             exclude_params={"New lesion": "rule 8", "Non-target lesion": "rule 9"}),
        dict(file="demo.sas7bdat", kind="wide", subject="SUBJID", variables=amgen_demo({
            "B_BMI": V("pre", "baseline", "Baseline BMI"), "B_LDHNM": V("pre", "baseline", "Baseline LDH"), "B_ALPNM": V("pre", "baseline", "Baseline alkaline phosphatase"),
            "B_CEA": V("pre", "baseline", "Baseline CEA"), "B_CREAT": V("pre", "baseline", "Baseline creatinine")}),
             exclude={"BIOPTPCT": "rule 9 biopsy percentage categories", "TRTDUR": "rule 8 exposure"}),
    ],
))

TRIALS.append(dict(
    id="Colorec_Amgen_2006_309", source="PDS", nct="NCT00364013",
    title="Amgen 20050203 (PRIME): panitumumab plus FOLFOX vs FOLFOX, first-line metastatic colorectal",
    design="parallel", unit="individual", status="included",
    arms=dict(file="adsl_pds2019.sas7bdat", subject="SUBJID", column="TRT",
              evidence="DDT_203: TRT = randomized treatment; ATRT = actual"),
    tables=[
        dict(file="adlb_pds2019.sas7bdat", kind="long", subject="SUBJID", param="LBTEST", value="LBSTRESN", visit="VISIT", role="safety_lab",
             visits=dict(pre=["Screening"], pre_dose=["Cycle 1, day 1"], post=CYCLE_DAY), param_roles={"Carcinoembryonic Antigen": "secondary"}),
        dict(file="adls_pds2019.sas7bdat", kind="long", subject="SUBJID", param="LSCAT", value="LSSLD", visit="VISIT", role="tumour",
             visits=dict(pre=["Screening"], pre_dose=[], post=WEEK),
             exclude_params={"New lesion": "rule 8", "Non-target lesion": "rule 9"}),
        dict(file="adpm_pds2019.sas7bdat", kind="wide_visit", subject="SUBJID", visit="VISIT", role="vital",
             visits=dict(pre=["Screening"], pre_dose=["Cycle 1, day 1"], post=CYCLE_DAY),
             variables={"WEIGHT": V("post", "vital", "Body weight by visit (adpm)", "WEIGHT")}),
        dict(file="adsl_pds2019.sas7bdat", kind="wide", subject="SUBJID", variables={
            "AGE": V("pre", "baseline", "Age"), "B_WEIGHT": V("pre", "baseline", "Baseline weight"), "B_HEIGHT": V("pre", "baseline", "Baseline height"), "DIAGMONS": V("pre", "baseline", "Months since diagnosis")},
             exclude={"DTHDY": "rule 8 censored", "PFSDYCR": "rule 8 censored"}),
    ],
))

TRIALS.append(dict(
    id="HeadNe_Amgen_2007_265", source="PDS", nct="NCT00460265",
    title="Amgen 20050251 (SPECTRUM): panitumumab plus chemotherapy vs chemotherapy, recurrent or metastatic head and neck",
    design="parallel", unit="individual", status="included",
    arms=dict(file="demo.sas7bdat", subject="SUBJID", column="TRT",
              evidence="DDT: TRT = randomized treatment (TRTCD 2/50); ATRT = actual"),
    tables=[
        dict(file="lab.sas7bdat", kind="long", subject="SUBJID", param="LBTESTCD", value="LBSTRESN", visit="VISIT", role="safety_lab",
             visits=dict(pre=["Screening"], pre_dose=["Cycle 1, day 1"], post=CYCLE_DAY), exclude_params=AMGEN_LAB_EXCL),
        dict(file="lesion.sas7bdat", kind="long", subject="SUBJID", param="LSCAT", value="LSSLD", visit="VISIT", role="tumour",
             visits=dict(pre=["Screening"], pre_dose=[], post=WEEK),
             exclude_params={"Non-target lesion": "rule 9"}),
        dict(file="demo.sas7bdat", kind="wide", subject="SUBJID", variables=amgen_demo(),
             exclude={"BIOPTPCT": "rule 9", "DXMNFDOS": "rule 8 dose", "MXMNFDOS": "rule 8 dose", "TRTDUR": "rule 8"}),
    ],
))

TRIALS.append(dict(
    id="LungSm_Allianc_1998_261", source="PDS", nct="NCT00003299",
    title="CALGB 9732: paclitaxel added to cisplatin/etoposide, extensive small-cell lung cancer",
    design="parallel", unit="individual", status="negative_control_only",
    arms=dict(file="c9732_demographic.sas7bdat", subject="PHATOM_ID", column="TRT_ARM",
              evidence="Dictionary: TRT_ARM 1 = cisplatin,etoposide; 2 = paclitaxel,cisplatin,etoposide,G-CSF"),
    tables=[dict(file="c9732_demographic.sas7bdat", kind="wide", subject="PHATOM_ID", variables={"AGE": V("pre", "baseline", "Age")},
                 exclude={"OS_TIME": "rule 8", "PD_TIME": "rule 8", "PFS_TIME": "rule 8"})],
    notes="Survival trial; only age is continuous and pre-randomization.",
))

TRIALS.append(dict(
    id="LungSm_Allianc_2007_218", source="PDS", nct="NCT00486941",
    title="CALGB 30504: maintenance sunitinib vs placebo, extensive small-cell lung cancer",
    status="excluded", why="rule 8: only censored times (survtime, failtime, pfs2) and cycles of maintenance (exposure); no continuous pre-randomization variable (age is categorical)",
))

# Amgen 20010145 (darbepoetin in SCLC): laboratory tables are one row per visit with WEEK numeric.
NESP_LAB_VISITS = dict(pre_max=0, pre_dose=[1], post_min=2)
TRIALS.append(dict(
    id="LungSm_Amgen_2002_266", source="PDS", nct="NCT00119613",
    title="Amgen 20010145: darbepoetin alfa vs placebo with chemotherapy, extensive small-cell lung cancer",
    design="parallel", unit="individual", status="included",
    arms=dict(file="c_keyvar.sas7bdat", subject="SUBJID", column="TXGROUP",
              evidence="DDT: TXGROUP = randomized treatment group (NESP/PLACEBO); SAFGROUP = treatment actually received"),
    tables=[
        dict(file="c_hemat.sas7bdat", kind="wide_visit", subject="SUBJID", visit="WEEK", visits=NESP_LAB_VISITS, role="safety_lab",
             variables={k: V("post", "safety_lab", f"Hematology {k} by study week (c_hemat)", k) for k in ["RBC", "HGB", "HCT", "PLT", "WBC", "ANC", "NEUT", "EOSIN", "SEGS", "BANDS"]}),
        dict(file="c_chem.sas7bdat", kind="wide_visit", subject="SUBJID", visit="WEEK", visits=NESP_LAB_VISITS, role="safety_lab",
             variables={k: V("post", "safety_lab", f"Chemistry {k} by study week (c_chem)", k) for k in ["SODIUM", "POTAS", "CL", "BICARB", "TLPROT", "ALBUMN", "GLUC", "BUN", "UREA", "CREAT", "URACID", "TLBILI", "ALKPH", "LDH", "SGOT", "SGPT"]}),
        dict(file="c_vitals.sas7bdat", kind="wide_visit", subject="SUBJID", visit="WEEK", visits=NESP_LAB_VISITS, role="vital",
             variables={k: V("post", "vital", f"Vital sign {k} by study week (c_vitals)", k) for k in ["SBP", "DBP", "PULSE"]}),
        dict(file="c_lesion.sas7bdat", kind="wide_visit", subject="SUBJID", visit="WEEK", visits=NESP_LAB_VISITS, role="tumour",
             variables={"CXSINDEX": V("post", "tumour", "Chest X-ray target lesion sum (DDT)", "CXSINDEX"), "CTSINDEX": V("post", "tumour", "CT scan target lesion sum (DDT)", "CTSINDEX")}),
        dict(file="a_qol.sas7bdat", kind="wide", subject="SUBJID", variables={
            **{f"{s}B": V("pre", "pro", f"FACT {s} baseline score", s, 0) for s in ["FAT", "FGT", "FGF", "FGP", "FGE", "FGS", "HSI", "VAS", "FAN"]},
            **{f"{s}W{w}": V("post", "pro", f"FACT {s} score at week {w}", s, i + 1) for s in ["FAT", "FGT", "FGF", "FGP", "FGE", "FGS", "HSI", "VAS", "FAN"] for i, w in enumerate([7, 13, 24, 50])},
        }, exclude={"*CHG*": "rule 8 change scores", "*EOS": "rule 12 end-of-study visit"}),
        dict(file="c_bchar.sas7bdat", kind="wide", subject="SUBJID", variables={
            "AGE": V("pre", "baseline", "Age"), "B_HGB": V("pre", "baseline", "Baseline hemoglobin"), "B_SEREPO": V("pre", "baseline", "Baseline serum erythropoietin"),
            "B_WEIGHT": V("pre", "baseline", "Baseline weight"), "B_HEIGHT": V("pre", "baseline", "Baseline height")}),
    ],
    notes="Primary endpoints were change in hemoglobin and transfusion; hemoglobin by week is the primary-endpoint quantity. c_iron is drawn at study day <= 1 only (pre_dose) and is not used.",
))

TRIALS.append(dict(
    id="LungSm_EliLill_2011_287", source="PDS", nct="NCT01439568",
    title="Lilly LY2510924 plus carboplatin/etoposide vs carboplatin/etoposide, extensive small-cell lung cancer (phase 2)",
    design="parallel", unit="individual", status="included",
    arms=dict(file="dm.sas7bdat", subject="USUBJID", column="ARMCD", exclude_levels=["SCRNFAIL"], filter={"RANDFL": "Y"},
              evidence="SDTM DM: ARMCD planned arm; RANDFL = randomized flag; SCRNFAIL never randomized"),
    tables=[
        dict(file="hema_cen.sas7bdat", kind="long", subject="USUBJID", param="LBTEST", value="SIRESN", visit="VISIT", role="safety_lab",
             visits=dict(pre=["Baseline"], pre_dose=["Cycle 1"], post=["Cycle 1 Day 7", "Cycle 2", "Cycle 3"]),
             exclude_params={"RBC Morphology": "rule 9 text result"}),
        dict(file="chem_cen.sas7bdat", kind="long", subject="USUBJID", param="LBTEST", value="SIRESN", visit="VISIT", role="safety_lab",
             visits=dict(pre=["Baseline"], pre_dose=["Cycle 1"], post=["Cycle 1 Day 7", "Cycle 2", "Cycle 3"])),
        dict(file="vs.sas7bdat", kind="long", subject="USUBJID", param="VSTESTCD", value="VSSTRESN", visit="VISIT", role="vital",
             visits=dict(pre=[], pre_dose=["PRE-TREATMENT", "CYCLE 1"], post=["CYCLE 2", "CYCLE 3", "CYCLE 4", "CYCLE 5", "CYCLE 6"]),
             exclude_params={"ECOG": "rule 9", "HGT": "height is not an outcome"}),
        dict(file="tr.sas7bdat", kind="long", subject="USUBJID", param="TRTESTCD", value="TRORRES", visit="VISIT", role="tumour",
             visits=dict(pre=[], pre_dose=["Pre-Treatment"], post=["Cycle 3", "Cycle 5"]),
             exclude_params={"LDIAM": "per-lesion diameter (rule 8 duplicate of the sum)", "TUMSTATE": "rule 9 categorical"}),
        dict(file="dm.sas7bdat", kind="wide", subject="USUBJID", variables={"AGE": V("pre", "baseline", "Age")}),
    ],
    notes="The 'Pre-Treatment' visit precedes dosing but its relation to randomization is not stated, so it is pre_dose and unused (rule 7). Screening labs carry LBBLFL in lbcen but that file has no visit labels; hema_cen/chem_cen 'Baseline' is used instead.",
))

TRIALS.append(dict(
    id="Multiple_Allianc_2002_202", source="PDS", nct="NCT00028431",
    title="NCCTG N99C4: nicotine inhaler, bupropion, or both for smoking cessation (first randomization)",
    design="multiarm", unit="individual", status="included",
    arms=dict(file="casecrse.sas7bdat", subject="mask_id", column="ARM",
              evidence="DDT: ARM = first-randomization arm A/B/C; CURR_ARM reflects the second randomization at week 12 and is not used"),
    tables=[
        dict(file="visit.sas7bdat", kind="wide_windows", subject="mask_id", time="eval_time", filter={"FORM": 6}, role="clinical",
             windows=dict(pre=[None, 0], post=[["week 4", 21, 42], ["week 8", 42, 70], ["week 12", 70, 98]]),
             variables={"CO": V("post", "primary", "Expired-air carbon monoxide (ppm), the biochemical confirmation of abstinence", "CO"),
                        "CIGARETS": V("post", "secondary", "Average daily cigarettes, past 7 days", "CIGARETS"),
                        "SYS_PRES": V("post", "vital", "Systolic blood pressure", "SYS_PRES"), "DIA_PRES": V("post", "vital", "Diastolic blood pressure", "DIA_PRES"),
                        "PULSE": V("post", "vital", "Pulse", "PULSE"), "bmi": V("post", "vital", "BMI", "bmi")}),
        dict(file="smokehx.sas7bdat", kind="wide", subject="mask_id", variables={
            "AGE1": V("pre", "baseline", "Age started smoking (history)"), "CIGPD": V("pre", "baseline", "Cigarettes per day at entry"), "SMK_YRS": V("pre", "baseline", "Years smoked"), "NUMSTOP": V("pre", "baseline", "Number of prior quit attempts")}),
    ],
    notes="eval_time = days from first randomization to visit (DDT). Windows follow the protocol schedule (weeks 4, 8, 12 in phase I). First row per participant in a window is used.",
))

TRIALS.append(dict(
    id="Multiple_Allianc_2002_213", source="PDS", nct="NCT00052910",
    title="CALGB 80101: adjuvant chemoradiation regimens after gastric resection",
    design="parallel", unit="individual", status="negative_control_only",
    arms=dict(file="NCT00052910_D1_(EFFICACY).csv", subject="MASK_ID", column="TREAT_ASSIGNED",
              evidence="Data dictionary: TREAT_ASSIGNED 1/2 = randomized arm"),
    tables=[dict(file="NCT00052910_D1_(EFFICACY).csv", kind="wide", subject="MASK_ID", variables={
        "num_lymph_node_examined": V("pre", "baseline", "Nodes examined at the resection that preceded randomization"),
        "num_pos_lymph_node": V("pre", "baseline", "Positive nodes at resection")},
        exclude={"os_year": "rule 8", "DFS_year": "rule 8", "STRATUM_GRP_ID": "rule 8 code"})],
))

TRIALS.append(dict(
    id="Multiple_Brigham_454", source="PDS", nct="NCT01169259",
    title="VITAL: vitamin D and omega-3 fatty acids, 2x2 factorial; knee pain ancillary study",
    design="factorial", unit="individual", status="included",
    arms=dict(file="VITAL_trial_NEJM_2022.sas7bdat", subject="Subject_ID", type="factorial",
              factors={"vitdactive": "vitamin D (1) vs placebo (0)", "fishoilactive": "omega-3 (1) vs placebo (0)"},
              evidence="Data dictionary: vitdactive, fishoilactive = randomization to active"),
    tables=[
        dict(file="vitalkneepain111623.sas7bdat", kind="wide", subject="Subject_ID", variables={
            **{f"{s}_base": V("pre", "pro", f"WOMAC {s} score at baseline (dictionary)", s, 0) for s in ["pain", "function", "stiffness"]},
            **{f"{s}_yr{y}": V("post", "pro", f"WOMAC {s} score at year {y}", s, y) for s in ["pain", "function", "stiffness"] for y in [1, 2, 3, 4]},
        }),
        dict(file="VITAL_trial_NEJM_2022.sas7bdat", kind="wide", subject="Subject_ID", variables={
            "bmi": V("pre", "baseline", "BMI at randomization"), "ageyr": V("pre", "baseline", "Age at randomization")},
             exclude={"*years": "rule 8 censored event times", "*yrs": "rule 8"}),
    ],
    notes="hsCRP and VitD_total (baseline biomarkers) are keyed by newid, which does not link to Subject_ID; excluded. Knee pain cohort is 1,398 of 25,871.",
))

CLOVIS_POST = r"^Cycle (\d+) (?:1[ab] )?\(Day (\d+)\)$"
TRIALS.append(dict(
    id="Pancrea_ClovisO_2010_186", source="PDS", nct="NCT01124786",
    title="Clovis CO-101-001 (LEAP): CO-101 (gemcitabine elaidate) vs gemcitabine, metastatic pancreatic cancer",
    design="parallel", unit="individual", status="included",
    arms=dict(file="adsl.sas7bdat", subject="USUBJID", column="TRT01P",
              evidence="ADaM ADSL: TRT01P planned treatment; TRT01A actual"),
    tables=[
        dict(file="adlb.sas7bdat", kind="long", subject="USUBJID", param="PARAMCD", value="AVAL", visit="AVISIT", role="safety_lab",
             visits=dict(pre=["Screening", "Baseline"], pre_dose=["Cycle 1 1a (Day 1)"], post=CLOVIS_POST),
             param_roles={"CA19_9AG": "secondary"},
             exclude_params={k: "rule 13 or morphology flags" for k in ["NEUTB", "NEUTBLE", "METAMY", "METAMYLE", "MYCY", "MYCYLE", "HYPO", "MACROCY", "RBCNUC", "ANISO", "LYMAT", "LYMATLE", "POIKILO", "POLYCHR", "TARGCELL", "ROULEAUX", "EGFRCG", "ELLIPCY", "BLAST", "BLASTLE", "PLATGNT", "ACANT", "TOXGRAN"]}),
        dict(file="advs.sas7bdat", kind="long", subject="USUBJID", param="PARAMCD", value="AVAL", visit="AVISIT", role="vital",
             visits=dict(pre=["Screening", "Baseline"], pre_dose=["Cycle 1 1a (Day 1)"], post=CLOVIS_POST),
             exclude_params={"HEIGHT": "not an outcome"}),
        dict(file="adeg.sas7bdat", kind="long", subject="USUBJID", param="PARAMCD", value="AVAL", visit="AVISIT", role="safety_lab",
             visits=dict(pre=["Baseline"], pre_dose=["Cycle 1 1a (Day 1)"], post=["Cycle 1 1b (Day 29)"])),
        dict(file="adqs.sas7bdat", kind="long", subject="USUBJID", param="PARAMCD", value="AVAL", visit="AVISIT", role="pro",
             visits=dict(pre=["Screening", "Baseline"], pre_dose=["Cycle 1 1a (Day 1)"], post=CLOVIS_POST),
             exclude_params={k: "rule 9 single EQ-5D item" for k in ["EQ5D0101", "EQ5D0102", "EQ5D0103", "EQ5D0104", "EQ5D0105"]}),
        dict(file="adsl.sas7bdat", kind="wide", subject="USUBJID", variables={
            "AGE": V("pre", "baseline", "Age"), "HEIGHT": V("pre", "baseline", "Baseline height"), "WEIGHT": V("pre", "baseline", "Baseline weight"),
            "BSA": V("pre", "baseline", "Baseline BSA"), "BCA199": V("pre", "baseline", "Baseline CA19-9")},
             exclude={"DEATHDY": "rule 8", "RANDDUR": "rule 8", "DIAGDUR": "rule 8 derived duration", "METDDUR": "rule 8", "DGMTDUR": "rule 8"}),
    ],
))

# ============================================================================
# DRYAD
# ============================================================================
def dryad(doi, title, **kw):
    d = dict(id=f"dryad_{doi}", doi=f"10.5061/dryad.{doi}", source="Dryad", title=title)
    d.update(kw)
    return d

TRIALS.append(dryad("18931zdb7", "CHESS: multifaceted digital intervention for hypertension in primary care",
                    status="excluded", why="rule 3: open-label CLUSTER randomized trial (deposit title and abstract); participant-level tests invalid"))

TRIALS.append(dryad("275d4", "Panigrahi 2017: randomized synbiotic trial to prevent sepsis in rural Indian newborns",
    design="parallel", unit="individual", status="included",
    arms=dict(file="doi_10.5061_dryad.275d4__Panigrahi et al.SEPSSIF.CSV", subject="Study_ID", column="TX",
              evidence="TX 1/0 = synbiotic/placebo, 2,278 each, matching the published 1:1 allocation of 4,556 infants"),
    tables=[dict(file="doi_10.5061_dryad.275d4__Panigrahi et al.SEPSSIF.CSV", kind="wide", subject="Study_ID", variables={
        "wt7": V("post", "secondary", "Weight at day 7 (g); dosing began at enrolment (days 2-4)", "weight", 1),
        "Weight_Day_28": V("post", "secondary", "Weight at day 28 (g)", "weight", 2),
        "wt60": V("post", "secondary", "Weight at day 60 (g)", "weight", 3),
        "Birth_weight": V("pre", "baseline", "Birth weight (g)"),
        "Weight_Enroll": V("pre", "baseline", "Weight at enrolment (g)"),
        "MOMAGE": V("pre", "baseline", "Maternal age"),
    }, exclude={"BWT": "rule 8 duplicate of Birth_weight", "WTDIFF": "rule 8 change score", "Total_Doses": "rule 8 exposure", "Treat_time": "rule 8 exposure", "TIME": "rule 8 censored time to sepsis"})],
    notes="Study_ID has 4,554 unique values for 4,556 rows; duplicated ids are dropped at the first-record rule. Primary endpoint was sepsis or death (binary)."))

TERECO = "doi_10.5061_dryad.59zw3r27n__TERECO_Public_Data_wide.csv"
TRIALS.append(dryad("59zw3r27n", "TERECO: telerehabilitation for COVID-19 survivors, 1:1 block-randomized",
    design="parallel", unit="individual", status="included",
    arms=dict(file=TERECO, subject="num_ID", column="group_randomized",
              evidence="Column group_randomized (1/2) matches the published 59 vs 61 allocation; treat_corticosteroid and treat_oxygen are prior treatments, not allocation"),
    tables=[dict(file=TERECO, kind="wide", subject="num_ID", variables={
        **{f"_6MWD{i}": V(t, "primary", f"6-minute walking distance, assessment {i} ({lab})", "6MWD", o) for i, t, lab, o in [(1, "pre", "baseline", 0), (2, "post", "6 weeks, primary endpoint time", 1), (3, "post", "28 weeks", 2)]},
        **{f"{v}{i}": V(t, "secondary", f"{lab2}, assessment {i} ({lab})", v, o) for v, lab2 in [("squat", "Squat time (s)"), ("fvc", "FVC"), ("fev", "FEV1"), ("mvv", "Maximal voluntary ventilation"), ("pef", "Peak expiratory flow"), ("SF12_PCS", "SF-12 physical component"), ("SF12_MCS", "SF-12 mental component")]
          for i, t, lab, o in [(1, "pre", "baseline", 0), (2, "post", "6 weeks", 1), (3, "post", "28 weeks", 2)]},
        "age": V("pre", "baseline", "Age"), "los_days": V("pre", "baseline", "Hospital length of stay before enrolment"),
        "days_onset_to_admission": V("pre", "baseline", "Days from symptom onset to admission"),
    }, exclude={"*GLI*": "rule 8 rescaling of the same spirometry as percent predicted", "*_cn_*": "rule 8 rescaling", "fev_fvc_ratio*": "rule 8 derived ratio", "mMRC*": "rule 9 ordinal", "time_onset*": "pre-randomization timing variables, not outcomes"})],
))

TRIALS.append(dryad("8cz8w9gsn", "Insulin sensitivity in mesolimbic pathways: 3-month caloric restriction vs active waiting group, randomized",
    design="parallel", unit="individual", status="included",
    arms=dict(file="doi_10.5061_dryad.8cz8w9gsn__characteristics_behavioral_data.csv", subject="ID", column="group", read_kw={"encoding": "utf-8-sig"},
              evidence="README: group = diet group / waiting group; abstract: 'randomization to a 3-month caloric restriction or active waiting group'"),
    tables=[dict(file="doi_10.5061_dryad.8cz8w9gsn__characteristics_behavioral_data.csv", kind="wide", subject="ID", read_kw={"encoding": "utf-8-sig"}, variables={
        "BMI_t0": V("pre", "baseline", "BMI at t0 (README)", "BMI", 0), "BMI_t1": V("post", "primary", "BMI at t1 follow-up", "BMI", 1),
        "Homa2_PL_t0": V("pre", "baseline", "HOMA2 at t0", "Homa2", 0), "Homa2_PL_t1": V("post", "secondary", "HOMA2 at t1", "Homa2", 1),
        "HSminLS_liking_PL_t0": V("pre", "baseline", "High minus low sugar liking, placebo session t0", "HSminLS_PL", 0), "HSminLS_liking_PL_t1": V("post", "secondary", "same at t1", "HSminLS_PL", 1),
        "HSminLS_liking_IN_t0": V("pre", "baseline", "High minus low sugar liking, insulin session t0", "HSminLS_IN", 0), "HSminLS_liking_IN_t1": V("post", "secondary", "same at t1", "HSminLS_IN", 1),
        "fasting_time_PL_t0": V("pre", "baseline", "Fasting time before placebo session t0"), "fasting_time_IN_t0": V("pre", "baseline", "Fasting time before insulin session t0"),
        "food_liking_PL_t0": V("pre", "baseline", "Mean food liking t0"), "nonfood_liking_PL_t0": V("pre", "baseline", "Mean nonfood liking t0"),
    }, exclude={"weight_change_t0_minus_t1_kg": "rule 8 change score", "BMI_diff_t0t1_perc": "rule 8 change score", "hunger_*": "rule 9 0-10 rating", "VTA_BOLD*": "imaging contrast at t0 only"})],
))

TRIALS.append(dryad("9cnp5hqtv", "Snoezelen intervention for problem behaviours in cerebral palsy",
                    status="excluded", why="rule 13: 15 participants per arm"))

TRIALS.append(dryad("cv37539", "Cowling: randomized placebo-controlled trial of inactivated influenza vaccine in children (HAI titer mediation)",
    design="parallel", unit="individual", status="included",
    arms=dict(file="doi_10.5061_dryad.cv37539__kid_all.csv", subject="hhID", column="intervention",
              evidence="README: intervention = vaccination group assigned (TIV / placebo)"),
    tables=[dict(file="doi_10.5061_dryad.cv37539__kid_all.csv", kind="wide", subject="hhID", variables={
        "prevax.B.Brisbane": V("pre", "baseline", "Pre-vaccination HAI titer (9 distinct dilutions; fails rule 9)", "titer", 0),
        "postvax.B.Brisbane": V("post", "primary", "Post-vaccination HAI titer, 30 days (dilution series, 10 distinct)", "titer", 1),
        "age": V("pre", "baseline", "Age in years"),
    }, exclude={"surv_time": "rule 8 censored", "vaccine": "received vaccine (actual), not allocation"})],
    notes="Titers are a two-fold dilution series; the conventional analysis log2-transforms them, which is exactly a scale choice."))

TRIALS.append(dryad("k3j9kd5qd", "IV ferumoxytol vs oral ferrous sulfate for iron-deficiency anaemia in pregnancy",
    design="parallel", unit="individual", status="included",
    arms=dict(file="doi_10.5061_dryad.k3j9kd5qd__Deidentifed_Data4.11.26.csv", subject="ID", column="arm",
              evidence="Column arm = Oral iron / IV iron (ferumoxytol), 40 each, matching the abstract"),
    tables=[dict(file="doi_10.5061_dryad.k3j9kd5qd__Deidentifed_Data4.11.26.csv", kind="wide", subject="ID", variables={
        "screening_hgb": V("pre", "baseline", "Hemoglobin at screening", "hgb", 0),
        "fourwk_hgb": V("post", "primary", "Hemoglobin at week 4 (primary endpoint time)", "hgb", 1),
        "eightwk_hgb": V("post", "secondary", "Hemoglobin at week 8", "hgb", 2),
        "hgb_admit": V("post", "secondary", "Hemoglobin at delivery admission", "hgb_admit", 1),
        "ferritin": V("pre", "baseline", "Ferritin at screening"), "tsat": V("pre", "baseline", "Transferrin saturation at screening"),
    }, exclude={"ga_deliv_wks": "rule 9 (9 distinct)", "parity": "rule 9"})],
))

CLIMB = "doi_10.5061_dryad.m0cfxpp37__CLIMB_AnalysisDataset_BMJOpen_v4.csv"
TRIALS.append(dryad("m0cfxpp37", "CLIMB: chlorhexidine oral rinses vs placebo for symptomatic COPD, randomized 1:1",
    design="parallel", unit="individual", status="included",
    arms=dict(file=CLIMB, subject="subject_id", column="randassign",
              evidence="Column randassign = active / placebo (24 / 20); microbial *_group_* columns are taxa, not allocation"),
    tables=[dict(file=CLIMB, kind="wide", subject="subject_id", variables={
        **{f"sgrq_{s}_score_visit1": V("pre", "baseline", f"SGRQ {s} at visit 1 (baseline)", f"sgrq_{s}", 0) for s in ["symptoms", "activity", "impacts", "total"]},
        **{f"sgrq_{s}_score_visit2": V("post", "secondary", f"SGRQ {s} at visit 2 (week 8)", f"sgrq_{s}", 1) for s in ["symptoms", "activity", "impacts", "total"]},
        "biomass_norm_ow1": V("pre", "baseline", "Oral wash 16S biomass, normalized, baseline", "biomass_ow", 0),
        "biomass_norm_ow2": V("post", "primary", "Oral wash 16S biomass, normalized, week 8 (primary endpoint)", "biomass_ow", 1),
        "biomass_norm_spt1": V("pre", "baseline", "Sputum biomass baseline", "biomass_spt", 0),
        "biomass_norm_spt2": V("post", "primary", "Sputum biomass week 8", "biomass_spt", 1),
        "shannon_ow1": V("pre", "baseline", "Oral Shannon diversity baseline", "shannon_ow", 0), "shannon_ow2": V("post", "secondary", "Oral Shannon diversity week 8", "shannon_ow", 1),
        "simpson_ow1": V("pre", "baseline", "Oral Simpson diversity baseline", "simpson_ow", 0), "simpson_ow2": V("post", "secondary", "Oral Simpson diversity week 8", "simpson_ow", 1),
        "shannon_spt1": V("pre", "baseline", "Sputum Shannon baseline", "shannon_spt", 0), "shannon_spt2": V("post", "secondary", "Sputum Shannon week 8", "shannon_spt", 1),
        "crp": V("pre", "baseline", "CRP at baseline (crp_changew8 exists, so crp is the baseline value)"),
        "fibrinogen": V("pre", "baseline", "Fibrinogen at baseline"), "leukocyte": V("pre", "baseline", "Leukocytes at baseline"),
        "fev1_predict": V("pre", "baseline", "FEV1 percent predicted at baseline"), "fvc_predict": V("pre", "baseline", "FVC percent predicted at baseline"),
        "pack_years": V("pre", "baseline", "Pack-years"), "years_smoke": V("pre", "baseline", "Years smoked"),
    }, exclude={"*_chang*": "rule 8 change scores", "*_chg*": "rule 8", "biomass_raw_*": "rule 8 duplicate of normalized", "mass_sample_*": "process", "*_ow1": "taxon abundances, rule 9 (mostly zero)", "*_spt*": "taxon abundances"})],
))

TRIALS.append(dryad("n8pk0p31r", "Maternal education trial in rural Uganda, 8-year follow-up",
                    status="excluded", why="rule 3: cluster-randomized (README: two-armed open cluster randomized trial)"))
TRIALS.append(dryad("pc866t1t6", "Two-way SMS breastfeeding support, Kenya",
                    status="excluded", why="rules 5 and 9: outcomes are binary or categorical (exclusive breastfeeding, challenges); the only continuous variables are message counts (process) and income (baseline, categorical bands)"))
TRIALS.append(dryad("qnk98sfm2", "Community health worker home visits for child development, South Africa",
                    status="excluded", why="rule 3: cluster-randomized (title); clusterid has 51 levels"))

NT = "doi_10.5061_dryad.rg57q6m__Song_ntrainerdeidentified_datadryad_2.13.19.csv"
TRIALS.append(dryad("rg57q6m", "NTrainer: patterned oral stimulation in preterm infants, multicentre randomized",
    design="parallel", unit="individual", status="included",
    arms=dict(file=NT, subject=["siteid", "studyid"], column="Intervention Group",
              evidence="Column 'Intervention Group' = NTrainer Group / Control Group (109/101, matching the abstract); SubGroup is a gestational-age stratum"),
    tables=[dict(file=NT, kind="wide", subject=["siteid", "studyid"], variables={
        "growthvelocity": V("post", "secondary", "Weight growth velocity (g/kg/day) over the stay", "growthvelocity", 1),
        "ofcgrowth": V("post", "secondary", "Head circumference growth (cm/week)", "ofcgrowth", 1),
        "on CPAP": V("post", "clinical", "Days on CPAP during the stay", "on CPAP", 1),
        "on NC": V("post", "clinical", "Days on nasal cannula", "on NC", 1),
        "on Room Air": V("post", "clinical", "Days on room air", "on Room Air", 1),
        "bw": V("pre", "baseline", "Birth weight"), "ofc": V("pre", "baseline", "Head circumference at birth"), "ga": V("pre", "baseline", "Gestational age at birth"),
    }, exclude={"PMAfirstPO": "rule 8 censored (Censoredpo flag)", "pmafof": "rule 8 censored time to full oral feeds", "pmadischarge": "rule 8 censored by transfer/death", "los": "rule 8 censored (Censoredlos flag)", "Time to FOF": "rule 8 censored",
                "PMAfirstintervention": "rule 8 process", "interventionnumber": "rule 8 process", "interventionscompleted": "rule 8 process", "percentinterventionscompleted": "rule 8 process", "protocoldeviationpercent": "rule 8 process",
                "percent*": "rule 8 duplicate of day counts as percentages"})],
    notes="studyid is a site-local number: three ids recur at different sites, so the participant key is (siteid, studyid), 210 unique. Primary endpoint (time to full oral feeds) is censored and excluded."))

WL = "doi_10.5061_dryad.sj3tx96k5__"
TRIALS.append(dryad("sj3tx96k5", "Tailored multichannel weight-loss intervention in adolescents, randomized",
    design="parallel", unit="individual", status="included",
    arms=dict(file=WL + "baseline_assessment.csv", subject="code ", column="group",
              evidence="group 1/2 in the baseline file (133/146); the deposit's information sheet describes randomization to intervention and control"),
    tables=[
        dict(file=WL + "baseline_assessment.csv", kind="wide", subject="code ", variables={
            "weight": V("pre", "baseline", "Weight at baseline", "weight", 0), "BMI": V("pre", "baseline", "BMI at baseline", "BMI", 0), "height": V("pre", "baseline", "Height at baseline")}),
        dict(file=WL + "2_months_assessment.csv", kind="wide", subject="code ", variables={
            "weight": V("post", "primary", "Weight at 2 months", "weight", 1), "BMI": V("post", "primary", "BMI at 2 months", "BMI", 1)}),
        dict(file=WL + "5_months_assessment.csv", kind="wide", subject="code ", variables={
            "weight": V("post", "primary", "Weight at 5 months (end of intervention)", "weight", 2), "BMI": V("post", "primary", "BMI at 5 months", "BMI", 2)}),
    ],
    notes="Questionnaire items are 5-point Likert and fail rule 9."))

TRIALS.append(dryad("sqv9s4ncs", "KAT-AKI: kidney action team recommendations for hospitalized AKI, randomized",
                    status="excluded", why="rules 5 and 8: outcomes are binary (primary_outcome, dialysis, death) or counts of recommendations delivered (process); baseline_creat and start2enrollment_days are the only continuous variables and both are pre-randomization, so the deposit would contribute negative controls only; retained in the ledger as negative_control_only",
                    design="parallel", unit="individual", status_override="negative_control_only",
                    arms=dict(file="doi_10.5061_dryad.sqv9s4ncs__KAT-AKI_limited_dataset.04-23-26.csv", subject=None, column="randomization",
                              evidence="Column randomization 0/1 (2,004 / 1,999)"),
                    tables=[dict(file="doi_10.5061_dryad.sqv9s4ncs__KAT-AKI_limited_dataset.04-23-26.csv", kind="wide", subject=None, variables={
                        "baseline_creat": V("pre", "baseline", "Baseline creatinine"), "age_int": V("pre", "baseline", "Age (integer)")},
                        exclude={"*_count*": "rule 8 process", "total_recs*": "rule 8 process"})]))


def main() -> None:
    for t in TRIALS:
        if t.get("status_override"):
            t["status"] = t.pop("status_override")
        path = OUT / f"{t['id']}.yaml"
        with open(path, "w") as fh:
            yaml.safe_dump(t, fh, sort_keys=False, allow_unicode=True, width=120)
    print(f"wrote {len(TRIALS)} manifests to {OUT}")
    from collections import Counter
    print(Counter(t["status"] for t in TRIALS))


if __name__ == "__main__":
    main()
