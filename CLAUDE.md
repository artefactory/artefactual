# Artefactual Project Guide

## Commands
- Lint/Format: `ruff check` or `ruff format`
- Run single test: `python path/to/test.py`
- Run script with config: `python scripts/path/to/script.py --config.param=value`

## Code Style
- **Typing**: Strict typing with `beartype` runtime checks and `@override` decorators
- **Imports**: No relative imports, sorted with isort (standard lib → third-party → first-party)
- **Classes**: Use dataclasses with `@dataclasses.dataclass` and `@edc.dataclass` decorators
- **Config**: Inherit from `Serializable` for CLI parsing
- **Naming**: PascalCase for classes, snake_case for functions/variables, UPPER_CASE for constants
- **Line length**: 120 chars (80 for docstring code examples)
- **Documentation**: Google-style docstrings with triple double-quotes
- **Error handling**: Descriptive exceptions with f-strings
- **Validation**: Use annotation from the `beartype.vale` module

## Architecture
- Artefactual is a library - scripts must import and use it as an independent library
- Always refactor reusable code into the artefactual library when possible
- Configuration classes in `src/artefactual/config/`
- Scripts use `eapp.make_flags_parser` for CLI arguments
- Use `eapp.better_logging()` for improved logging setup
- All dependencies and configurations must be defined in pyproject.toml
