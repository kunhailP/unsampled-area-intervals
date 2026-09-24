PY ?= python3
PROCS ?= 8

.PHONY: test data quick tables apipop conditional theory paper all

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

# exact shrink function, small-noise map, heterogeneous noise, HetLDC (hours on ~15 cores)
theory: data
	$(PY) experiments/e16_exact_shrink_table.py $(PROCS)
	$(PY) experiments/e18_hetero_kernel.py $(PROCS)
	$(PY) experiments/e19_boundary_map.py $(PROCS)
	$(PY) experiments/fig_boundary_map.py
	$(PY) experiments/e21_hetldc_apipop.py 120 $(PROCS)
	$(PY) experiments/e22_estimated_variance.py 120 $(PROCS)
	$(PY) experiments/e23_allocation.py $(PROCS)
	$(PY) experiments/e24_certified_table.py $(PROCS)
	$(PY) experiments/e24_certified_table.py $(PROCS) 5e-4 0.002 0.0095 0.9,0.9036
	$(PY) experiments/e24_certified_table.py $(PROCS) 5e-4 0.002 0.366 0.9068
	$(PY) experiments/e20_hetldc_synth.py 150 $(PROCS)
	$(PY) experiments/e25_certified_ldc.py 2000 $(PROCS)
	$(PY) experiments/e27_interval_constants.py $(PROCS)
	$(PY) experiments/e28_level_decomposition.py
	$(PY) experiments/e29_estimated_scale.py 100 $(PROCS)
	$(PY) experiments/e30_areawise_variance.py 30 $(PROCS)

all: quick tables apipop conditional theory
	$(PY) experiments/e02_design_mixing.py
	$(PY) experiments/e06_synthetic_mc.py
	$(PY) experiments/e11_pums_structure.py ri

# needs a TeX engine; TEX = tectonic -X compile (single binary) or latexmk -pdf
TEX ?= tectonic -X compile
paper:
	$(PY) experiments/fig_paper.py
	cd paper && $(TEX) main.tex
