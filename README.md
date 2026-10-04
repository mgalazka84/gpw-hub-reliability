# GPW hub reliability

Code and derived research outputs for **Hub reliability and diversification risk in Polish stock correlation networks**, by Marek Gałązka and Hanna Wdowicka. Version **v1.0.1** corresponds to the manuscript prepared for submission to the *International Review of Financial Analysis*. This does not imply acceptance or publication by the journal.

The study asks whether a stock's stable network position adds information about subsequent diversification loss after accounting for observed concentration and standard financial predictors. It combines exact covariance diagnostics, controlled simulations and observed Polish stock prices.

## Main observed findings

The historical WIG20/mWIG40 application contains **155 analysed stocks**, **231 estimation origins during 2007–2025**, and **73 complete retrospective test windows during 2019–2025**. Each network uses 252 past sessions and 500 joint block-bootstrap replications. No missing market value is imputed.

* At least one hub has selection frequency at least 0.80 in 129 raw-network origins, five demeaned origins and eight WIG-residual origins.
* Adding WIG-residual reliability to financial predictors and observed concentration increases primary test MSE by **2.20%**. The paired 95% block interval includes zero.
* The open-to-close sensitivity estimate is a **2.69% MSE reduction**, also inconclusive.

These results do not establish incremental forecasting value. They do not identify causal shock transmission or validate an investable trading strategy.

## Reproduce saved derived results offline

Use Python 3.12. Pinned versions in `requirements.txt` match the study's numerical environment.

```bash
python -m venv .venv
# Activate the virtual environment using your operating system's command.
python -m pip install -r requirements.txt
python reproduce.py
```

The default command needs **no raw prices and no network access** after dependencies are installed. It runs method checks, verifies all saved hub concentration and reliability values, reproduces chronological test predictions from matured labels, checks test MSE and paired intervals, recreates empirical figures, and builds the editable manuscript in `final/deliverables/`. Convert the DOCX to PDF with Word or LibreOffice.

This is reproduction from published **derived inputs**. It does not independently re-collect or validate raw prices. The original full price-level verification record is `empirical/results/observed_verification.json`; the default command writes its separate, explicitly narrower report to `public_verification.json`.

## Rebuild from separately obtained prices

Raw quotations, price/return panels, GPW portfolio PDFs and their full extractions are excluded. Their public redistribution rights have not been established. Consult the original providers' terms before acquiring or reusing source inputs. The code uses ordinary public requests and does not bypass login, payment or access restrictions.

```bash
python tools/acquire_data.py
python reproduce.py --from-prices
```

Acquisition downloads recorded GPW portfolio URLs, obtains available company and WIG histories, and prepares local panels. It requires `curl` and Poppler's `pdftotext` on the command path. The second command rebuilds all empirical networks and forecasts and is substantially slower than the default check.

The manuscript uses the download vintage identified by retrieval timestamps and hashes in the acquisition manifests, with observations capped at **19 September 2025**. Providers can revise histories or remove series. A later download may not reproduce that exact vintage or its hashes. Reacquisition can overwrite tracked outputs and acquisition logs locally; retain the v1.0.1 tag as the manuscript reference. Historical hashes and source URLs are in the acquisition manifests and audit summary.

## Data scope and limitations

* Histories were available for 164 of 171 identities; 155 stocks passed past-only eligibility in at least one window. Ended histories and former constituents are included where available.
* Membership uses quarterly snapshots strictly before each origin. Extraordinary replacements and exact announcement timing are not fully reconciled.
* Prices are used as supplied. Denomination changes were diagnosed, but a complete corporate-action and cash-dividend audit was not available. Neither convention is certified as dividend-inclusive total return.
* Missing future returns for any selected stock make the entire target missing in both conventions. There are 42 omitted targets overall, including 10 of 83 test-period origins. Evaluation uses 73 complete test windows.
* The single WIG factor, overlapping estimation windows, retrospective design and short realized-covariance windows limit inference. There is no claim of prospective preregistration.

## Repository contents

| Directory | Contents |
| --- | --- |
| `submission/` | Network and forecast methods, method checks, simulation code and outputs, recorded constituent-source metadata. The name is retained from the original project. |
| `empirical/` | Acquisition/preparation/estimation scripts, derived features, node and edge estimates, predictions, settings and audit summaries. |
| `final/` | Manuscript builder and descriptive firm/volatility summaries. |
| `tools/` | Acquisition orchestration and verification using public derived outputs. |

Original Monte Carlo runs can be repeated with `submission/run_simulations.py` and `submission/variance_sensitivity.py`; consult their `--help` options. `submission/verify_reproduction.py` regenerates all 15 first-sample simulation examples and checks frozen production-source hashes.

## License and citation

Original code, documentation and derived research outputs provided here are released under the MIT license. It grants no rights over separately obtained third-party quotations or portfolio documents. Source URLs and hashes identify inputs, not a license to redistribute them.

Use `CITATION.cff` and **v1.0.1** when citing the manuscript's software version:

> Gałązka, M., & Wdowicka, H. (2026). *GPW hub reliability: Code and derived research outputs* (Version v1.0.1) [Computer software]. GitHub. https://github.com/mgalazka84/gpw-hub-reliability/tree/v1.0.1

ChatGPT (OpenAI) was used to assist with Python coding and data acquisition and extraction.
