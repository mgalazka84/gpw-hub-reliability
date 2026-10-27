# Degree inference with independent selection

This extension implements and evaluates finite-sample inference selected on
independent training observations. It contains 6,600 validation datasets,
20 matched procedures (132,000 outcome records), and an ordinal questionnaire
application. The archived pilot is separate from validation.

The featured selected procedure, FS-T64, uses 20% of observations to select a
candidate tree, its path comparisons, their fixed testing order and betting
fractions. The remaining observations supply verification evidence averaged
over 64 matchings. Testing stops at the first failed comparison. Conditional
FWER control uses the classical fixed-sequence argument. A64 implements
predictable verification-only betting; S64 uses the fixed quarter stake.
Neither the fixed-sequence principle nor e-value averaging is claimed as a
new generic testing theorem.

At p=12 and n=1,008, FS-T64 certifies the unique hub in 211/300 Gaussian-star
and 120/300 t5-star datasets, compared with 0/300 for E64. The asymptotic
multiplier comparator reaches 300/300 in both cells. All short-sample and
near-tie failures are retained. The selected exact procedure has boundary-cell
false-order rates 0.001, 0.009, 0.001 and 0.000 (1,000 replications each).

The adult Rosenberg application has 35,079 complete records, 7,015 used for
selection and 28,064 for verification. FS-T64 fixes eight of ten degrees and
certifies Q1, Q3 and Q10 as shared maximum-degree members. Both compatible
trees have maximum degree three. E64 retains an additional tree with a unique
degree-four hub. These are tau-a association statements under an IID working
model, not clinical or causal conclusions.

## Reproduce

Use Python 3.12, NumPy, pandas, SciPy, Matplotlib and a C++17 compiler. Run from
the repository root. Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before
starting Python. The scripts compile their small C++ helpers automatically.

```bash
python contrast_inference/power_revision/fetch_rse.py
python contrast_inference/power_revision/verify_methods.py
python contrast_inference/power_revision/validate.py --workers 6
python contrast_inference/power_revision/multiplier_validation.py
python contrast_inference/power_revision/application.py
python contrast_inference/power_revision/audit_application.py
python contrast_inference/power_revision/build_results.py
```

Validation is resumable through per-dataset checkpoints. The compressed
replication tables are the complete archived results; generated checkpoint
JSON files need not be redistributed. Simulations retain the same reference
models as the baseline package. `review_audit/` contains the independently
checked exhaustive Abalone analysis and its supplied relations.

## Contents and interpretation

- `methods.py`, `adaptive_betting.cpp`: selected and adaptive procedures.
- `PROTOCOL.txt`, `results/validation/design.json`: choices and frozen seeds.
- `results/validation/replications.csv.gz`: all 19 exact-variant outcomes.
- `results/multiplier_validation/replications.csv.gz`: matched M-C outcomes.
- `results/report_numbers.json`: integer counts and Wilson intervals.
- `results/application/`: provenance, all node bounds, relations, compatible
  trees, strong-polytope certificates, and independent verification.
- `manuscript/`: main and supplementary sources and exact scientific figures.

Each procedure is evaluated at its own alpha=.05. Do not pool conclusions from
multiple procedures or choose the most favourable verification result without
an additional error allocation. A fraction clipped below at .01 does not
inherit the fixed-family E64 consistency theorem for arbitrarily small gaps.

The archived internal specification records the dataset and processing rules.
Protocol and results entered the public archive together, so the repository does
not independently establish prospective specification or pre-application primary
method designation. No external preregistration was made; see PROVENANCE_AUDIT.txt. The all-complete-ages analysis changes
the target population and is reported as sensitivity analysis. The original
source describes anonymous research-use consent; the Item Response Warehouse
lists its release as CC BY 4.0. Original-source records are downloaded by hash
and are not included here. Repeated response profiles are not used to infer
repeated participation. Ties affect tau-a; tau-b point-tree sensitivity has no
inherited tau-a confidence guarantee.

Main numerical tables report validation and application outcomes. The
supplement retains earlier baseline experiments and timings as separate
analyses. No large-graph scaling or full-U empirical-Bernstein implementation
is claimed by this extension.

## Random-split sensitivity

The post-application check in `split_stability.py` retains all 100 additional
splits and 15 split variants. FS-T64 certifies the three shared members in 16/100
runs, with median six fixed degrees (range one to eight); the displayed split is
favourable. In 61/100 runs, all bounds and hub memberships are at least as
informative as the original full-sample E64 result. Enumeration agrees with the
hub decisions in all 100 runs. All per-run results and definitions are archived
in `results/split_stability/`; `verify_split_stability.py` checks aggregation and
replays three specified runs independently. Each procedure has its own nominal
guarantee, not simultaneous coverage across all variants and random splits.
