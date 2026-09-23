# unsampled-area-intervals

Research code and working notes on **prediction intervals for the realised mean of an area
that the survey did not sample**, when the only calibration data are noisy survey estimates
(with estimated design variances) for the areas that were sampled.

**Status: exploratory research, not a finished method.** The current state of every claim —
confirmed, numerical only, already in the literature, or refuted — is in
[`docs/FINDINGS.md`](docs/FINDINGS.md). Read that first. The notes are in Korean.

## What is here

| Piece | Where | Status (see FINDINGS) |
|---|---|---|
| Exact two-stage SRSWOR observation kernel on small binary finite populations; rank of the observation operator; a two-world counterexample in which areas with <= 3 sampled PSUs cannot distinguish latent distributions that need very different interval widths | `src/uai/kernel.py`, `experiments/e01` | confirmed |
| Mixing two-stage designs across areas identifies more moments, but did not shorten the identified interval | `e02` | numerical, small model |
| Log-concave quantile/sd constants (truncated-Laplace extremal) and the worst-case coverage of a normal interval (87.75% at 90%) | `src/uai/logconcave.py`, `e03` | **known** for symmetric log-concave laws (Barthe–Koldobsky; He–Tkocz–Wyczesany 2023, Lemma 3) |
| Log-concave deconvolution shrink factor r_alpha(D/t^2): from a noisy conformal threshold to a latent-target threshold | `logconcave.py`, `e04`, `results/r_table.json` | numerical; novelty undetermined |
| Exact marginal latent-coverage guarantee of split conformal rules via the Beta law of the conformal order statistic | `e05`, `results/exact_coverage.csv` | Beta law exact; worst-case H numerical |
| Synthetic Monte Carlo; apipop finite-population comparisons with noisy conformal and Fay–Herriot (REML and parametric bootstrap) | `e06`–`e08` | FH met nominal coverage on apipop and was 12–17% narrower than the certified rule |
| Negative checks (multi-quantile constraints, depth allocation); ACS PUMS structure check | `e09`–`e11` | numerical |

## Reproduce

Python >= 3.10.

```bash
pip install -e ".[test]"
make test          # fast checks of the core numbers
make data          # downloads apipop from the CRAN `survey` package into data/ (not redistributed)
make quick         # E01, E03, E05 (from the stored tables), E10: minutes
make tables PROCS=64   # E04, E05 compute, E09: global searches, long
make apipop PROCS=64   # E07, E08: 1,000 replications each
```

Run scripts from the repository root. Stored outputs are in `results/`. Per-replication
pickles and downloaded data go to `data/` (git-ignored).

## Layout

```
src/uai/        kernel.py  logconcave.py  procedures.py
experiments/    e01 ... e11, fetch_apipop.py, _common.py
results/        small JSON/CSV outputs of the experiments
docs/           FINDINGS.md (current state)  HISTORY.md  RESEARCH_DESIGN.md (dated log)  START_GATE_MEMO.md
tests/          test_core.py
```

## Data

- `apipop` (California Academic Performance Index population, via the R `survey` package), fetched by
  `experiments/fetch_apipop.py`.
- ACS 2024 1-year PUMS person file for one state, fetched by `e11`. Treated as a fixed record set; its
  means are not population truths for US residents.
