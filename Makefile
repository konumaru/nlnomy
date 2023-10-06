default: help

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: run
run: # Run the streamlit app.
	poetry run streamlit run app.py

.PHONY: test
test: # Run the tests.
	poetry run pytest tests

.PHONY: requirements
requirements: # Update the requirements.txt file.
	poetry export --without-hashes -f requirements.txt -o requirements.txt
