# Contributing to cunei-tools

We welcome contributions from both NLP researchers and cuneiform scholars.

## For Cuneiform Scholars

The most valuable contribution you can make is **improving the sign dictionary**. If you encounter unmatched signs or incorrect mappings:

### Adding Sign Corrections

1. **Export unmatched signs** from your corpus:
   ```python
   from cunei_tools import CuneiConv
   conv = CuneiConv.load_from_file("sign_dict.json")
   conv.export_unmatched(your_texts, "unmatched.csv")
   ```

2. **Fill in the Unicode column** in the exported CSV using your expertise.

3. **Submit your corrections** as a pull request adding your CSV to the `corrections/` directory, or email them to us.

### Supported Correction Formats

You can provide corrections in any of these formats:

| Format | Example |
|--------|---------|
| CSV | `sign,unicode` columns |
| TSV | Tab-separated sign/unicode pairs |
| JSON | `{"sign_name": "unicode_char"}` |
| TXT | One `sign<TAB>unicode` pair per line |
| Excel | First two columns as sign/unicode |

All corrections are attributed. Include your name and affiliation in the `source` column so we can credit your work.

## For NLP Researchers

### Adding New Features

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for your changes
4. Run tests: `python -m pytest tests/ -v`
5. Submit a pull request

### Code Style

- Follow PEP 8
- Add docstrings to all public methods
- Include type hints where practical
- Write tests for new functionality

### Adding Language Support

To add support for a new cuneiform language:

1. Determine the optimal segmentation threshold using `CuneiSeg.find_optimal_threshold()`
2. Add the threshold to `LANG_THRESHOLDS` in `segmenter.py`
3. Add any language-specific normalization to `converter.py`
4. Include test cases with real examples from the language
5. Document the data source and any special handling needed

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All tests must pass before submitting a pull request.

## Reporting Issues

Please include:
- Python version
- cunei-tools version
- Minimal code example reproducing the issue
- For sign mapping issues: the transliteration input and expected Unicode output
