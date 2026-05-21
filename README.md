# ML Project Template

## Structure Overview

### src/
Core source code.

- **data/**: Data loading and preprocessing
- **models/**: Model definitions
- **training/**: Training logic
- **evaluation/**: Metrics and validation
- **utils/**: Helper functions

### configs/
Configuration files (YAML). Avoid hardcoding parameters in code.

### notebooks/
Exploration only. Do NOT put production logic here.

### main.py
Entry point of the pipeline.

---

## Philosophy

- Keep components modular
- Avoid hardcoding (use config)
- Separate responsibilities
- Make everything reusable
