venv-python: ## Install Python 3 venv
	python -m venv venv

venv-dev: venv-python ## Install Python 3 dev dependencies
	./venv/bin/pip install -r dev-requirements.txt

venv-dev-upgrade: ## Upgrade Python 3 dev dependencies
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install --upgrade -r dev-requirements.txt

venv: venv-python ## Install Python 3 run-time dependencies
	./venv/bin/pip install -r requirements.txt

venv-upgrade: ## Upgrade Python 3 run-time dependencies
	./venv/bin/pip install --upgrade -r requirements.txt

# ===================================================================
# Linters and profilers
# ===================================================================

format: venv-dev-upgrade ## Format the code
	@git ls-files '*.py' | xargs ./venv/bin/python -m autoflake --in-place --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --exclude="compat.py,globals.py"
	./venv/bin/python -m black ./mgclass --exclude outputs/static

# ===================================================================
# Docs
# ===================================================================

docs: venv-dev-upgrade ## Create the documentation
	jupyter nbconvert --output-dir="./docs/experiments" --to markdown "./experiments/*.ipynb"
	./venv/bin/python -m mkdocs build

docs-server: venv-dev-upgrade ## Start a Web server to serve the documentation
	jupyter nbconvert --output-dir="./docs/experiments" --to markdown "./experiments/*.ipynb"
	./venv/bin/python -m mkdocs serve

release-note: ## Generate release note
	git --no-pager log $(LASTTAG)..HEAD --first-parent --pretty=format:"* %s"
	@echo "\n"
	git --no-pager shortlog -s -n $(LASTTAG)..HEAD

# ===================================================================
# Run
# ===================================================================

run:
	./venv/bin/python -m mgclass