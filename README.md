# Mosaic

A photo mosaic generator that creates mosaics from Google Photos.

## Project Structure

```
mosaic/
├── mosaic/              # Main package directory
│   ├── __init__.py      # Package initialization
│   ├── lib.py           # Core library functions and Thumb class
│   ├── download_images.py  # Download images from Google Photos
│   ├── deduper.py       # Find and handle duplicate images
│   ├── images.py        # Main mosaic generation logic
│   └── blend.py         # Image blending utilities
└── tests/               # Unit tests
    ├── __init__.py
    └── test_lib.py      # Tests for lib module
```

## Usage

### Running scripts as modules

```bash
# Download images from Google Photos
python -m mosaic.download_images

# Find duplicate images
python -m mosaic.deduper

# Generate a mosaic
python -m mosaic.images

# Blend images
python -m mosaic.blend
```

### Importing in code

```python
from mosaic.lib import Thumb, read_thumbs
from mosaic import download_images, deduper, images, blend
```

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_lib

# Run with verbose output
python -m unittest tests.test_lib -v
```

## Development

This project uses:
- **black** for code formatting
- **isort** for import sorting

Format code before committing:
```bash
black mosaic/ tests/
isort mosaic/ tests/ --profile black
```

### Optional: Pre-commit hooks

To automatically format code before each commit, install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

The pre-commit configuration will automatically run black and isort on staged files.
