# Simultaneous degree and rank inference for correlation-tree hubs

Methodological extension prepared for author review before a possible submission to **Statistical Papers**, by **Marek Gałązka and Hanna Wdowicka**. This directory does not imply journal submission, acceptance or peer-reviewed validation.

The method projects a simultaneous band for edge weights through maximum spanning trees. Two lexicographic tree computations per vertex give exact coordinatewise degree endpoints over the rectangular weight region. The rank intervals are conservative outer bounds. Exactness over the box does not imply a sharp projection onto valid correlation matrices.

The original financial application and its version `v1.0.1` remain separate. The extension addresses population degree and rank uncertainty, rather than forecasting improvements.

## Reproduce and check

From the repository root, with the pinned root dependencies installed:

```bash
python statistical_papers/verify_envelopes.py
python statistical_papers/verify_finite_band.py
python statistical_papers/verify_saved_results.py
python statistical_papers/build_paper.py
cd statistical_papers/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error SP_manuscript.tex
```

The first three commands check the projection against exhaustive labelled-tree enumeration, compare tied-data Kendall tau-a with its direct definition, check every saved aggregate, and regenerate one complete bootstrap sample per simulation cell. The manuscript is built from saved numerical outputs. A LaTeX distribution with `latexmk` and BibTeX is required. Springer Nature's unmodified class and bibliography style are included with their original notices; those third-party files retain their original licenses and are not relicensed under the repository's MIT license.

Full simulation runs:

```bash
python statistical_papers/run_simulations.py
python statistical_papers/run_simulations.py --models two_hubs chain --output simulation_shapes
python statistical_papers/run_kendall.py
```

There are 7500 primary Monte Carlo data sets, 3000 additional non-star data sets, and 3000 independent Kendall-variant data sets. Bootstrap runs use 999 resamples each. Seeds and replication-level results are saved. For predictable execution time, setting `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` is useful on machines with many cores.

To regenerate the GPW extension, first acquire the original data as described in the root README. Then point to the empirical directory containing the locally held price panels and saved eligible universes:

```bash
python statistical_papers/run_gpw.py --data-root empirical
python statistical_papers/run_gpw.py --data-root empirical --block 5 --stride 5 --output gpw_block5
python statistical_papers/run_gpw.py --data-root empirical --block 20 --stride 5 --output gpw_block20
```

The original `empirical/results/hub_selections.csv.gz` is required to preserve the exact past-only eligible universe. Reacquisition may change the source vintage. The extension's input hashes are recorded in `results/gpw_design.json`. Raw third-party quotations and portfolio documents are not redistributed.

## What the guarantees cover

- The projection inherits coverage from the input joint band. The deterministic computation does not guarantee that a particular bootstrap band is calibrated.
- The Kendall concentration band has finite-sample simultaneous coverage under independent identically distributed observations, including ties. It is conservative and is not applied to serially dependent stock returns.
- The Pearson implementation uses a block-bootstrap band for centered Fisher-transformed correlations. Its guarantee is asymptotic under explicit moment, dependence and bootstrap-consistency assumptions. Finite-sample undercoverage of the band is reported, not hidden by high coverage of broad degree intervals.
- Population edge ties are handled by retaining all optimal trees. Degree ties retain every compatible rank resolution.
- GPW intervals are nominal and per window/representation. No simultaneous claim over time, no selective-inference guarantee for the eligibility filter, and no causal or trading interpretation is made.
- In all 231 primary GPW windows, the method leaves every rank possible for every vertex. The procedure is insufficiently precise for positive hub certification on these data.

The mathematical ingredients are connected explicitly to existing interval-weight optimization and rank-inference literature. No claim of priority for classical exchange arguments, resampling, or confidence-region projection is made.

ChatGPT (OpenAI) assisted with coding, data handling, methodological development and manuscript drafting. Authors remain responsible for the submitted work.
