# Finite-sample edge-order inference and certified tree degree comparisons

Marek Gałązka and Hanna Wdowicka. Research version **0.5.3**.

The current article, implementation and complete validation results are in
[`contrast_inference/`](contrast_inference/README.md), on branch
[`csda-submission-v0.5.3`](https://github.com/mgalazka84/gpw-hub-reliability/tree/csda-submission-v0.5.3).

The revision combines valid Kendall tau-a matching e-values with certified
spanning-tree degree and rank projections. It includes eight matched methods,
7,800 simulated datasets, a separate 3,000-dataset Pearson-bootstrap audit,
local/complete family and projection ablations, practical bootstrap frequencies,
and 192 real measurement analyses. Original UCI morphometric data carry their
CC BY 4.0 attribution. Raw third-party stock-price histories are not redistributed.

The exact procedures require IID rows. Abalone results use an explicit IID
working model; Dry Bean is a computational stress study. The Polish market
analysis targets Pearson trees of dependent returns and is reported separately
as a sensitivity study. Bootstrap calibration and finite-sample guarantees are
distinguished throughout the article.

See [`contrast_inference/README.md`](contrast_inference/README.md) for verification
and recomputation, and [`contrast_inference/submission_materials/`](contrast_inference/submission_materials/)
for the paper, anonymous manuscript, title page and Highlights. `contrast_inference/MANIFEST.json` records unchanged-file SHA-256 hashes.

Earlier versions remain available at their original branches and tags, including
`empirical-revision-v0.4`. The other root directories preserve the earlier GPW
forecasting study; the current revision is self-contained in `contrast_inference/`.
Software authorship and version details are in `CITATION.cff`. Original code and
original derived outputs use the repository MIT licence; the data retain their
separately stated licences.
