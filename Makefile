SOURCES=src

.isort_fix:
	isort ${SOURCES}

.black_fix:
	black -q  ${SOURCES}

format: .isort_fix .black_fix

.check_repo:
	@if [ "$(CHECK_COMMIT)" != "false" ]; then \
		if [ -n "$$(git status --porcelain)" ]; then \
			echo "There are changes in the repo. Please commit the code in order to maintain traceability of the experiments"; \
			exit 1; \
		fi \
	fi

holdout:
	@$(MAKE) -s .check_repo
	@python src/evaluation/holdout_from_params.py &> stdout_log

holdout_report:
	@$(MAKE) -s .check_repo
	@python src/evaluation/holdout_from_report.py &> stdout_log

grid_search:
	@$(MAKE) -s .check_repo
	@python src/evaluation/grid_search.py &> stdout_log
