# Finite-sample edge-order inference and certified degree comparisons

Authors: Marek Gałązka and Hanna Wdowicka. Revision 0.3.

This directory replaces the scientific focus of the preceding manuscript. It contains exact conditional McNemar tests of Kendall edge orders using predetermined disjoint IID observation pairs; Holm familywise correction; graphic-minor degree-difference bounds; cycle-path LP restrictions with rational dual certificates; and sharp rank projection over integer degree boxes subject to the tree degree sum.

The finite-sample theorem assumes IID rows and independent tie randomization. It does **not** validate the Pearson block bootstrap for dependent financial returns. The bootstrap validation contains severe undercoverage, retained in full. Every GPW rank interval remains [1,p], despite tighter degree bounds. This revision is a research manuscript for author review; no new journal submission has been made.

## Reproduce

From repository root, with Python dependencies in `requirements.txt`, a C++17 compiler, and LaTeX for PDF generation:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python contrast_inference/verify_revision.py
python contrast_inference/verify_example.py
python contrast_inference/run_exact_simulations.py --repetitions 200
python contrast_inference/run_bootstrap_simulations.py --repetitions 100 --bootstrap 499
python contrast_inference/verify_revision.py --gpw
python contrast_inference/build_results.py
latexmk -pdf -cd contrast_inference/manuscript/revision.tex
```

The exact experiment resumes from its replication CSV. To intentionally reproduce it from scratch, move that CSV aside first. Seeds and target designs are stored in `results/*_design.json`. Do not mix partial files from experiments with different settings.

To regenerate the empirical statistics, supply the frozen, separately retained data assembly:

```bash
python contrast_inference/run_refined_gpw.py --data-root /path/to/empirical --bootstrap 999 --workers 4
```

Required inputs are `data/close_simple_returns.csv.gz`, `data/wig_ohlc.csv`, and `results/hub_selections.csv.gz`. Their hashes are in the GPW design file. Source acquisition and limitations are preserved in the earlier empirical release, tag `v1.0.1`. Third-party raw prices are not redistributed here. The supplied derived order relations and rational certificates allow independent verification of every improved LP endpoint without the raw input panel.

## Results and scope

- 4,400 IID exact-inference data sets, 200 repetitions per cell; full-sample Hoeffding baseline on the same data.
- 2,400 Pearson-bootstrap data sets, 100 repetitions per cell and 499 bootstrap draws, including serial dependence and dimensions 12 and 52.
- All 231 frozen GPW origins, three representations, 999 bootstrap draws per origin: 693 analyses and 35,979 vertex records.
- 85,129 improving rational certificates from the empirical LP projections.
- All finite small-instance checks and independent certificate verification are recorded as JSON.

Classical ingredients are credited in the manuscript, including McNemar, Holm, Zaffalon–Hutter, matroid optimization, and interval spanning-tree literature. No claim is made that cut/cycle screening, greedy matroid rank identities, or confidence-set projection are themselves new. The code demonstrates a specific statistical construction and stronger checked projections; publication suitability remains an editorial assessment.

`manuscript/revision.pdf` is the revised English manuscript; `REWIZJA_MERYTORYCZNA_PL.pdf` summarizes changes and limitations in Polish. The frozen previously submitted version remains in `statistical_papers/`.
