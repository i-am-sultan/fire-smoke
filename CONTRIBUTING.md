# 🤝 Contributing to Fire & Smoke Detection Pipeline

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## How to Contribute

### 1. Fork & Clone

```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR_USERNAME/fire-smoke-training-pipeline.git
cd fire-smoke-training-pipeline
git remote add upstream http://10.10.30.65:3000/trainee-ai-ml/fire-smoke-training-pipeline.git
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/new-augmentation-strategy`
- `fix/onnx-export-error`
- `docs/add-quickstart-guide`
- `test/add-integration-tests`

### 3. Make Changes

Follow the code guidelines below.

### 4. Test Your Changes

Before committing, verify your changes work:

```bash
# Test individual scripts
python scripts/dataset/standardize.py --help
python scripts/training/train.py --help

# Test the full pipeline (if applicable)
python pipeline.py -c pipeline.yaml
```

### 5. Commit & Push

```bash
git add .
git commit -m "feat: add new deduplication algorithm"
git push origin feature/your-feature-name
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring

### 6. Open a Pull Request

Submit a pull request on GitHub with:
- Clear title describing the change
- Description of what & why
- Any breaking changes
- Related issues (if applicable)

---

## Code Guidelines

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** on function signatures:
  ```python
  def process_image(image_path: Path, threshold: float = 0.5) -> dict:
      """Process image and return detections."""
      pass
  ```

- Add **docstrings** to functions and classes:
  ```python
  def clean_dataset(dataset_dir: Path, remove_corrupted: bool = True) -> int:
      """
      Clean dataset by removing corrupted images and invalid annotations.
      
      Args:
          dataset_dir: Path to dataset root directory
          remove_corrupted: Whether to remove corrupted files
          
      Returns:
          Number of files removed
      """
      pass
  ```

### Logging

- Use the standard logging module (not print statements)
- Follow the existing logging pattern in scripts:
  ```python
  import logging
  
  logger = logging.getLogger(__name__)
  logger.info("Processing started")
  logger.warning("Skipping corrupted file")
  logger.error("Failed to load model")
  ```

### Configuration

- Keep configuration in YAML files (pipeline.yaml)
- Avoid hardcoding values — use constants or config files
- Document all configuration parameters with comments

### Error Handling

- Use specific exceptions, not generic `Exception`
- Provide helpful error messages
- Log stack traces for debugging:
  ```python
  try:
      model.load_weights(weights_path)
  except FileNotFoundError:
      logger.error(f"Weights not found: {weights_path}")
      sys.exit(1)
  ```

---

## Testing

*Note: Testing infrastructure is planned for v2.0*

Currently, we perform manual testing:

1. **Unit testing**: Run individual scripts with `--help` to verify argument parsing
2. **Integration testing**: Run `python pipeline.py -c pipeline.yaml` with sample data
3. **Regression testing**: Test existing workflows after code changes

When contributing, please test:
- [ ] Script runs without errors: `python script.py --help`
- [ ] No breaking changes to existing APIs
- [ ] Output files are created in expected locations
- [ ] Logging is informative and error-free

---

## Documentation

- Update [README.md](README.md) if adding new features
- Update relevant docs in `docs/` directory
- Add docstrings to new functions
- Update [docs/pipeline.md](docs/pipeline.md) if adding pipeline stages

---

## Areas We Need Help With

### High Priority
- **Testing**: Unit tests & integration tests for all scripts
- **Error Handling**: Improve validation & error messages across scripts
- **Type Hints**: Add complete type annotations to all scripts
- **CI/CD**: GitHub Actions workflows for automated testing

### Medium Priority
- **Performance**: Optimize data loading, training speed
- **Documentation**: Tutorials, use-case examples, video guides
- **Deployment**: Additional export formats (TFLite, CoreML, ONNX with optimization)
- **Augmentation**: Domain-specific augmentation strategies

### Lower Priority
- **UI/Dashboard**: Web interface for monitoring training
- **Multi-language**: Support for non-English documentation
- **Model Zoo**: Pre-trained models for specific fire/smoke scenarios

---

## Code Review Process

- All pull requests require at least one approval
- We look for:
  - Code quality (style, readability, maintainability)
  - Functionality (does it work as described?)
  - Testing (was it tested?)
  - Documentation (is it documented?)
  - Performance (any regressions?)

---

## Questions?

- Check [docs/troubleshooting.md](docs/troubleshooting.md) for common issues
- Open a GitHub Issue for questions or bugs
- See [README.md](README.md#-support--issues) for support channels

---

## Code of Conduct

- Be respectful and inclusive
- Assume good intentions
- Constructive feedback only
- Welcome to all backgrounds and experience levels

---

## License

By contributing, you agree your code will be licensed under the same MIT License as the project.

Thank you for contributing! 🎉
