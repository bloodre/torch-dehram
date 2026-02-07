.PHONY: install install-dev clean build

# Basic knobs (override via environment when needed)
POETRY ?= poetry
MAKE ?= make
TORCH_VER ?= 2.8.0
CUDA ?= cu129
POETRY_DEV_GROUPS ?= dev

torch-deps:
	$(POETRY) run python -m pip install --upgrade pip
	$(POETRY) run python -m pip install \
		--index-url https://download.pytorch.org/whl/$(CUDA) \
		torch==$(TORCH_VER)
	$(POETRY) run python -m pip install \
		-f https://data.pyg.org/whl/torch-$(TORCH_VER)+$(CUDA).html \
		torch-scatter>=2.0.0 torch-sparse>=0.6.0
	rm -f "=0.6.0" "=2.0.0"

install:
	$(POETRY) install
	$(MAKE) torch-deps

# install-dev -> same flow as install, then pull extra groups (set POETRY_DEV_GROUPS if needed)
install-dev:
	$(POETRY) install --with $(POETRY_DEV_GROUPS)
	$(MAKE) torch-deps

# clean -> drop any poetry-managed virtualenvs for this project
clean:
	$(POETRY) env remove --all || true

# build -> produce distribution artifacts via poetry
build:
	$(POETRY) build