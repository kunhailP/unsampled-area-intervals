PY ?= python3
PROCS ?= 8

.PHONY: test data quick tables apipop conditional all

test:
	$(PY) -m pytest -q

data:
	$(PY) experiments/fetch_apipop.py

# minutes on a laptop
quick:
	$(PY) experiments/e01_start_gate.py
	$(PY) experiments/e03_lc_constants.py
	$(PY) experiments/e05_exact_coverage.py
	$(PY) experiments/e10_depth_allocation.py

# expensive searches: tens of minutes to hours depending on PROCS
tables:
	$(PY) experiments/e04_shrink_table.py $(PROCS)
	$(PY) experiments/e05_exact_coverage.py compute $(PROCS)
	$(PY) experiments/e09_multi_quantile.py

apipop: data
	$(PY) experiments/e07_apipop_ldc.py 1000 $(PROCS)
	$(PY) experiments/e08_apipop_fh.py 1000 $(PROCS)

# conditional reliability (E12-E14): about an hour on 100+ cores
conditional: data
	$(PY) experiments/e12_conditional_synth.py 2000 $(PROCS)
	$(PY) experiments/e13_conditional_apipop.py 1000 $(PROCS)
	$(PY) experiments/e14_grid_check.py $(PROCS)
	$(PY) experiments/e15_e12_exact.py $(PROCS)

all: quick tables apipop conditional
	$(PY) experiments/e02_design_mixing.py
	$(PY) experiments/e06_synthetic_mc.py
	$(PY) experiments/e11_pums_structure.py ri
