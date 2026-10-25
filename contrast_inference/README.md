# Exact edge-order inference and certified tree degree comparisons

Marek Gałązka and Hanna Wdowicka. Practical-value revision 0.6.

The manuscript studies population **Kendall tau-a** correlation spanning trees.
It combines exact disjoint-pair McNemar tests, dependence-safe p-value averaging,
and finite-sample evidence averaging over 8 or 64 matchings. Evidence products
retain original zero ties and include a sample-size-dependent betting fraction
for consistency. Full-U-statistic Hoeffding and Gaussian multiplier procedures
provide matched comparators. Graphic-minor, degree-conservation and cycle-path
LP projections yield integer degree and rank bounds. Improved LP endpoints have
independently checked rational weak-duality certificates.

The exact guarantees assume IID observations. Gaussian multiplier contrasts
show finite-size error inflation in several null cells; Pearson-bootstrap
alternatives also remain undercalibrated. The Polish-market analyses are a
separate sensitivity study of dependent returns and Pearson trees. Abalone
measurements illustrate exact Kendall inference under an explicit IID working
model. Dry Bean includes 68 identical rows and is a computational stress study.

## Practical benefit and computational cost

The main title is **Certified degree bounds and hub exclusions in correlation
spanning trees**. The revision evaluates changed conclusions separately from
narrower intervals. `certify_lp(ans, p, degree_pairs=False)` runs the degree-only
stage; the default `True` preserves the previous full ordered-pair projection.

- In pooled Abalone, holding E64 and the local family fixed, degree LP increases
  excluded hubs from 2 to 4, fixes a second leaf degree, and restricts whole weight
  from rank [1,6] to [1,3]. Full pair LP changes none of these endpoints.
- Degree-only LP preserves full rank endpoints in 188 of all 192 measurement
  analyses, and in all 12 E64 local-family populations. The remaining four
  analyses gain six exclusions from pair LP. Analyses overlap and are not
  independent replications of practical success.
- The controlled Abalone E64 projection takes 0.0391 versus 0.1534 seconds
  (16 versus 72 LP objectives). Calibration takes a separate 0.1400 seconds.
  This is a projection speedup, not the same speedup for the entire pipeline.
- All 50 timing cases use fresh serial processes, one pinned CPU, one thread,
  one warm-up and three measured repetitions per stage. Raw cases, timing runs,
  peak process memory and verified endpoints are supplied as compressed records.
- The same first 100 datasets are used on both sides of every p=12 Monte Carlo
  projection comparison. The audit retains all 27,200 method-family-dataset
  comparisons. Exact star hub power at n=1008 remains zero; chain degree-width
  gains often produce no changed decision. The Dry Bean E64 grid yields no leaf,
  hub exclusion or top-three certification despite narrower degree bounds.

These are retrospective results on the complete existing grid. No new statistical
power samples or broader real-world independence guarantees are claimed.

## Results

- 7,800 matched simulated datasets, 24 cells and eight methods. Every new null
  calibration cell has 1,000 replications, including complete edge families.
- Five matched projection stages. Path LP uses all degree-pair objectives on
  the first 100 p=12 datasets per cell; skipped sharp trivial cases are recorded.
- A separate 3,000-dataset Pearson audit, with nested 499/1,999 row resamples,
  two studentizers and Gaussian influence multipliers.
- Practical bootstrap frequencies on 2,800 matched simulation samples.
- 12 real measurement populations, eight methods and two families: 192 cases.
  All original UCI data, unchanged-file hashes and CC BY 4.0 attribution are supplied.
- 1,159 Polish-window/sensitivity analyses; the 109,548 previous improved
  rational certificates and source limitations are retained.

`results/comparative/replications_all.csv.gz` and `summary_all.csv` are the
canonical matched results. `results/review_report_numbers.json` contains the
numbers used in the manuscript; `results/review_verification.json` records final
checks. True unique-hub power is undefined for the chain/two-hub populations,
whose degree ties are retained.

## Verify the supplied results

Use Python 3.12, the packages in this directory's `requirements.txt`, a C++17
compiler, and a standard LaTeX installation for PDFs. From repository root:

```bash
python -m pip install -r contrast_inference/requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python contrast_inference/verify_comparators.py
python contrast_inference/verify_evalues.py
python contrast_inference/verify_estimands.py
python contrast_inference/verify_revision.py
python contrast_inference/verify_review.py
python contrast_inference/build_review_results.py --from-records
python contrast_inference/build_intro_figure.py
python contrast_inference/audit_practical_value.py
python contrast_inference/run_practical_benchmark.py --report
python contrast_inference/build_practical_results.py
python contrast_inference/build_practical_figure.py
latexmk -pdf -cd contrast_inference/manuscript/revision.tex
latexmk -pdf -cd contrast_inference/manuscript/anonymous.tex
```

The final verifier checks no projection failure on the valid-order event,
paired projection nesting, every null-cell replication count, original
measurement hashes, all saved measurement certificates with Python integers,
simple-path validity, reconstructed integer endpoints/ranks and matching of
the frequency-reference datasets. Its claim is computational verification;
sampling validity is established by the stated mathematical assumptions.
The `--from-records` command rebuilds the tables and figures from the published
individual outcomes without rerunning the simulations or retaining their caches.

## Recompute

```bash
python contrast_inference/fetch_cross_section.py
python contrast_inference/run_comparative_simulations.py --workers 4
python contrast_inference/run_family_null_extension.py --workers 4
python contrast_inference/run_evalue_validation.py --workers 4
python contrast_inference/run_bootstrap_audit.py --workers 4
python contrast_inference/run_frequency_reference.py --workers 4
python contrast_inference/run_cross_section.py --workers 2
python contrast_inference/run_evalue_cross_section.py --workers 2
python contrast_inference/build_review_results.py
python contrast_inference/verify_review.py
python contrast_inference/build_intro_figure.py
python contrast_inference/audit_practical_value.py
python contrast_inference/run_practical_benchmark.py --report
python contrast_inference/build_practical_results.py
python contrast_inference/build_practical_figure.py
latexmk -pdf -cd contrast_inference/manuscript/revision.tex
```

Cell checkpoints are local execution caches. Move the relevant cache directory
aside before changing a design; do not mix outputs from different algorithms or
settings. Complete final tables are built only after all required cells exist.
The main and evidence runs intentionally reuse the same simulated datasets;
they must not be counted as additional independent datasets. Frequency samples
are a subset of the main simulation. The work is retrospective method development.

To remeasure the cost grid on your own machine (the released times are hardware
specific), use `python contrast_inference/run_practical_benchmark.py --prepare --run`.
Move only the benchmark `runs/` cache aside first for a genuinely fresh timing run.
`--report` rebuilds summaries from the published compressed records without those
local caches. Timing repetitions are repeated computations of fixed instances.

## Historical market inputs

`run_refined_gpw.py` and `run_extended_gpw.py` require the separately retained
frozen market-price assembly. Raw third-party quotations are not redistributed.
Source records, hashes, derived relations, forward exclusions and rational
certificates are available. `verify_revision.py --gpw` checks base certificates;
`verify_extended_gpw.py` checks the extended source/selection/forward pipeline
when the original assembly is supplied. The data and bootstrap limitations are
stated in the article. Nonoverlapping returns alone do not establish IID sampling.

The mathematical foundations are credited: McNemar, Holm, bounded-variable
and U-statistic inequalities, e-value/p-value merging, robust tree inference,
matroid optimization, interval spanning trees and rank inference. The source
documents what is exact within an outer relaxation and what is asymptotic.

## Documents

`manuscript/revision.pdf` is the complete revised article. The submission package
also includes an anonymous manuscript, title page and Highlights. Dataset and software provenance are in
`data/cross_section/PROVENANCE.json`, `LICENSE_DATA.txt`, `CITATION.cff` and
`MANIFEST.json`. The 0.5.1 consistency check records separate local/complete LP counts and base/extended GPW counts. Earlier reproducibility records remain archived.

The introductory figure displays the pooled Abalone sample tree, 499/499
bootstrap unique-hub selections and the local E64 rank range [1,3].
`build_intro_figure.py` independently reconstructs the sample tau-a tree from
the raw measurements and checks its displayed values against frozen records.
Its record is `results/intro_figure_verification.json`; the simultaneous coverage
guarantee requires the stated IID working model.
