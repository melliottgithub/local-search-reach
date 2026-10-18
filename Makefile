# make setup   create .venv and install
# make run     run the three models (sample data unless data/private exists)
# make nb      execute both notebooks in place
# make clean   remove generated files

PY := .venv/bin/python

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

run:
	$(PY) src/reach.py
	$(PY) src/top3_classifier.py
	$(PY) src/synthetic_control.py

nb:
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_vetting_and_eda.ipynb
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

clean:
	rm -rf data/sample results/run

.PHONY: setup run nb clean
