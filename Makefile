# Reproduce the computed figures and numbers of the paper.
#
#   make all            figures and numbers.tex, from the files in results/
#   make figures        the six figures in figures/
#   make numbers        numbers.tex
#   make verify         rerun the checks of Section 5 (about two minutes)
#   make data           recompute the pgfplots tables (about fifteen minutes)
#   make train          retrain both networks of Section 5 (about six minutes)
#   make replot         redraw every figure from the cached results
#   make install PAPER=<path>
#                       copy the figures to <path>/figures/ and numbers.tex
#                       to <path>/
#   make clean          remove the LaTeX build products in tex/
#
# Files in results/ are committed.  A missing one is recomputed when a target
# needs it, an existing one only by make train, make verify or make data.

PYTHON ?= python3

LV_FIGS = figures/comparison_symmetric.png figures/comparison_asymmetric.png \
          figures/populations_symmetric.png figures/populations_asymmetric.png
PGF_FIGS = figures/heteroclinic-cycle.pdf figures/ring-cycle.pdf
FIGS = $(LV_FIGS) $(PGF_FIGS)

# Each script writes several files.  The one it writes last stands for all.
LV_TRAINING = results/lv-training.json
LV_VERIFY = results/lv-verify.json
RING = results/ring-results.json
HC = results/hc-orbit.dat

.PHONY: all figures numbers verify data train replot install clean

all: figures numbers

figures: $(FIGS)

numbers: numbers.tex

$(LV_FIGS): $(LV_TRAINING)
	$(PYTHON) train_lv.py --replot

figures/ring-cycle.pdf: tex/ring-cycle.tex $(RING)
	$(PYTHON) ring_cycle.py --replot

figures/heteroclinic-cycle.pdf: tex/heteroclinic-cycle.tex $(HC)
	$(PYTHON) heteroclinic_cycle.py --replot

numbers.tex: make_numbers.py $(LV_TRAINING) $(LV_VERIFY) $(RING)
	$(PYTHON) make_numbers.py

# Computations, run only when their output is missing.
$(LV_TRAINING):
	$(PYTHON) train_lv.py

$(LV_VERIFY): | $(LV_TRAINING)
	$(PYTHON) verify_lv.py

$(RING):
	$(PYTHON) ring_cycle.py

$(HC):
	$(PYTHON) heteroclinic_cycle.py

# Forced recomputations.
train:
	$(PYTHON) train_lv.py

verify:
	$(PYTHON) verify_lv.py

data:
	$(PYTHON) ring_cycle.py
	$(PYTHON) heteroclinic_cycle.py

replot:
	$(PYTHON) train_lv.py --replot
	$(PYTHON) ring_cycle.py --replot
	$(PYTHON) heteroclinic_cycle.py --replot

install: $(FIGS) numbers.tex
	@test -n "$(PAPER)" || { echo "usage: make install PAPER=<path>"; exit 1; }
	@test -d "$(PAPER)" || { echo "no such directory: $(PAPER)"; exit 1; }
	mkdir -p "$(PAPER)/figures"
	cp $(FIGS) "$(PAPER)/figures/"
	cp numbers.tex "$(PAPER)/"

clean:
	cd tex && latexmk -C ring-cycle.tex heteroclinic-cycle.tex
