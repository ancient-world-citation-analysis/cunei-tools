# cunei-tools

Cuneiform NLP utilities for word segmentation and script conversion across Akkadian, Sumerian, and Elamite.

Developed as part of *"Does Script Representation Matter? Evidence from Three Cuneiform Languages"* (Thompson & Anderson, EMNLP 2026).

## Installation

```bash
pip install cunei-tools
# Or from source:
git clone https://github.com/ancient-world-citation-analysis/cunei-tools
cd cunei-tools && pip install -e .
```

## cunei-seg: Word Boundary Segmentation

Recovers word boundaries from unsegmented Unicode cuneiform using transitional probability. Achieves **F1 > 0.96** across three languages.

| Language | F1 | Precision | Recall | Threshold |
|----------|------|-----------|--------|-----------|
| Akkadian | 0.969 | 0.946 | 0.992 | 0.70 |
| Sumerian | 0.971 | 0.946 | 0.998 | 0.60 |
| Elamite  | 0.973 | 0.993 | 0.955 | 0.90 |

### Python API

```python
from cunei_tools import CuneiSeg

# Train on segmented documents
seg = CuneiSeg(lang="akk")
seg.train([
    "𒀀𒈾 𒁀𒌅 𒊭",
    "𒂍 𒈾 𒀀𒈾",
    # ... your segmented corpus
])

# Segment new text
seg.segment("𒀀𒈾𒁀𒌅𒊭")
# → "𒀀𒈾 𒁀𒌅 𒊭"

# Find optimal threshold
metrics = seg.find_optimal_threshold(gold_documents)
print(metrics)
# → {'threshold': 0.70, 'f1': 0.969, 'precision': 0.946, 'recall': 0.992}

# Save and load
seg.save("akk_model.json")
seg2 = CuneiSeg.load("akk_model.json")
```

### Command Line

```bash
# Segment from file
cunei-seg --model akk_model.json --input unsegmented.txt --output segmented.txt

# Segment single text
cunei-seg --model akk_model.json --text "𒀀𒈾𒁀𒌅𒊭"
```

## cunei-conv: Script Conversion

Convert between Latin transliteration and Unicode cuneiform using 18,000+ sign mappings from Nuolenna and Akkademia.

### Python API

```python
from cunei_tools import CuneiConv

conv = CuneiConv()
conv.load_sign_lists()  # from GitHub

# Latin → Unicode
conv.to_unicode("a-na")        # → "𒀀 𒈾"
conv.to_unicode("šu-un-ki-ik") # → "𒋗 𒌦 𒆠 𒅅"

# Unicode → Latin
conv.to_latin("𒀀𒈾")          # → "a na"

# Detailed conversion with quality check
result = conv.to_unicode_detailed("a-na ba-x")
# → {'unicode': '𒀀 𒈾 𒁀 x', 'clean': False, 'unmatched': ['x']}

# Batch conversion rate
stats = conv.conversion_rate(["a-na", "ba-ab", "unknown-sign"])
# → {'total': 3, 'clean': 2, 'rate': 0.667, ...}

# Add manual corrections
conv.load_manual_corrections("corrections.csv")

# Save for offline use
conv.save("sign_dict.json")
conv2 = CuneiConv.load_from_file("sign_dict.json")
```

### Command Line

```bash
# Convert to Unicode
cunei-conv --mode to-unicode --text "a-na ba-ab"

# Convert file
cunei-conv --mode to-unicode --input latin.txt --output unicode.txt --lang akk

# Reverse: Unicode to Latin
cunei-conv --mode to-latin --text "𒀀𒈾"
```

## Supported Languages

- **Akkadian** (`akk`) — Semitic, ~1.25M tokens tested
- **Sumerian** (`sux`) — language isolate, ~146K tokens tested
- **Elamite** (`elx`) — isolating/agglutinative, ~13K tokens tested

The tools are language-agnostic at their core — they work with any cuneiform script that uses Unicode code points in the U+12000–U+1254F range. Language-specific parameters (thresholds, normalization) improve accuracy.

## Citation

```bibtex
@inproceedings{thompson-anderson-2026-script,
    title = "Does Script Representation Matter? Evidence from Three Cuneiform Languages",
    author = "Zhou, Chuanjun and Anderson, Adam",
    booktitle = "Proceedings of EMNLP 2026",
    year = "2026",
}
```

## Acknowledgments

The cuneiform-to-Unicode conversion pipeline builds on foundational work by Julia, Vinay, and Harrison Huang through the [CDLI2LoD](https://github.com/ancient-world-citation-analysis/CDLI2LoD) project at UC Berkeley's AWCA. Harrison has continued this work in evaluating and benchmarking conversion methods in his [Cdli2cuneiform](https://github.com/HarrisonLiruiHuang/Cdli2cuneiform) repository.

Sign list data comes from [Nuolenna](https://github.com/situx/Nuolenna) and [Akkademia](https://github.com/gaigutherz/Akkademia).


## Team

**Chuanjun Zhou** & **Adam Anderson**

## License

MIT