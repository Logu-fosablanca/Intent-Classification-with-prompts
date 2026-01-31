# How to Publish to PyPI

This guide explains how to upload your `query_classifier` package to the Python Package Index (PyPI) so others can install it with `pip install query-classifier`.

## Prerequisites

1.  **Create a PyPI Account**:
    - Go to [pypi.org](https://pypi.org/) and register.
    - (Optional but recommended) Go to [test.pypi.org](https://test.pypi.org/) and register there too, for testing.

2.  **Create an API Token**:
    - Go to **Account Settings** -> **API Tokens**.
    - Create a new token (scope: "Entire account" for now).
    - **Copy this token** (starts with `pypi-`). You will need it to log in.

## Step 1: Install Twine

Twine is the tool used to securely upload packages.

```bash
pip install twine
```

## Step 2: Build the Package

(You may have already done this. Check for a `dist/` folder.)

```bash
python -m build
```

This creates `.whl` and `.tar.gz` files in the `dist/` directory.

## Step 3: Check the Package (Recommended)

Verify that the distribution files are valid.

```bash
twine check dist/*
```

## Step 4: Upload to TestPyPI (Optional)

It's good practice to upload to the test server first to make sure everything looks right.

```bash
twine upload --repository testpypi dist/*
```
- **Username**: `__token__`
- **Password**: `pypi-<your-test-token>`

Then try installing it:
```bash
pip install --index-url https://test.pypi.org/simple/ query_classifier
```

## Step 5: Upload to PyPI (Production)

When you are ready to release to the world:

```bash
twine upload dist/*
```
- **Username**: `__token__`
- **Password**: `pypi-<your-production-token>`

## Success!

Your package will be live at `https://pypi.org/project/query_classifier/` (unless the name is taken; if so, change `name` in `setup.py`).

Users can now install it:
```bash
pip install query-classifier
```
