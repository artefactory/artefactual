.PHONY: examples

examples:
	uv run python examples/epr_usage_demo.py
	uv run jupyter lab examples/wepr_demo.ipynb
