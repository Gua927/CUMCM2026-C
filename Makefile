PYTHON ?= /root/miniconda3/bin/python

.PHONY: inspect prepare-results solve-q1 check-q1 solve-q2 check-q2 solve-q3 check-q3 figures test paper ai-details template-check check clean

figures:
	$(PYTHON) code/scripts/render_publication.py

solve-q2:
	$(PYTHON) code/scripts/solve_problem2.py $(Q2_ARGS)

check-q2:
	$(PYTHON) code/scripts/validate_problem2.py

solve-q3:
	$(PYTHON) code/scripts/solve_problem3.py

check-q3:
	$(PYTHON) code/scripts/validate_problem3.py

solve-q1:
	$(PYTHON) code/scripts/solve_problem1.py $(Q1_ARGS)

check-q1:
	$(PYTHON) code/scripts/validate_problem1.py

test:
	$(PYTHON) -m unittest discover -s code/tests -v

inspect:
	$(PYTHON) code/scripts/inspect_inputs.py

prepare-results:
	$(PYTHON) code/scripts/prepare_results.py --all

paper:
	bash code/scripts/build_paper.sh paper

ai-details:
	bash code/scripts/build_paper.sh ai-details

template-check: paper
	$(PYTHON) code/scripts/validate_submission.py --template output/paper/main.pdf

check: paper
	$(PYTHON) code/scripts/validate_submission.py output/paper/main.pdf

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; r=Path('output'); shutil.rmtree(r/'.build', ignore_errors=True); [p.unlink() for p in [r/'paper/main.pdf', r/'support/AI工具使用详情.pdf'] if p.exists()]"
