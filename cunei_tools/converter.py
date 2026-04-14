"""
cunei_conv: Latin transliteration ↔ Unicode cuneiform conversion.

Uses merged sign lists from Nuolenna + Akkademia (18,000+ mappings)
with optional manual corrections. Supports bidirectional conversion
and expert-driven dictionary updates.

Usage:
    conv = CuneiConv()

    # Load sign lists from GitHub (requires internet)
    conv.load_sign_lists()

    # Or load from local file
    conv.load_from_file("sign_dict.json")

    # Convert
    conv.to_unicode("šu-un-ki-ik")      # → "𒋗𒌦𒆠𒅅"
    conv.to_unicode("a-na")              # → "𒀀𒈾"
    conv.to_latin("𒀀𒈾")               # → "a na"

    # Batch convert
    conv.to_unicode_batch(["a-na", "šu-un-ki-ik"])

    # Check conversion quality
    result = conv.to_unicode_detailed("a-na ba-x")
    # → {'unicode': '𒀀𒈾 𒁀 x', 'clean': False, 'unmatched': ['x']}

    # === Expert dictionary updates ===

    # Add a single sign
    conv.add_sign("ŠU₂", "𒋗", source="Anderson 2026")

    # Load corrections from file (CSV, TSV, JSON, TXT, XLSX)
    conv.update_from_file("corrections.csv", source="CDLI corrections")

    # Export unmatched signs for experts to solve
    conv.export_unmatched(my_texts, "to_solve.csv")
    # → expert fills in the 'unicode' column
    conv.update_from_file("to_solve.csv", source="Expert review round 1")

    # Generate blank template
    conv.generate_template("new_signs.csv")

    # View update history
    conv.update_log()

    # Save (preserves all corrections and history)
    conv.save("sign_dict_updated.json")
"""

import json
import re
from pathlib import Path
from collections import OrderedDict


class CuneiConv:
    """Bidirectional Latin ↔ Unicode cuneiform converter."""

    def __init__(self):
        self.latin_to_uni = {}     # sign_name → unicode_char
        self.uni_to_latin = {}     # unicode_char → sign_name
        self.manual_corrections = {}
        self._update_log = []      # tracks all manual additions
        self.loaded = False

    def load_sign_lists(self):
        """
        Load and merge sign lists from GitHub (Nuolenna + Akkademia).
        Requires internet access.
        """
        import pandas as pd

        sign_list = pd.read_json(
            'https://raw.githubusercontent.com/situx/Nuolenna/master/sign_list.json',
            orient='index'
        )
        sign_list.columns = ['unicode']
        sign_list['sign'] = sign_list.index.tolist()
        sign_list = sign_list[['sign', 'unicode']].reset_index(drop=True)

        akkademia = pd.read_csv(
            'https://raw.githubusercontent.com/gaigutherz/Akkademia/master/'
            'cuneiform_to_unicode_fixed.csv'
        )
        merged = pd.merge(sign_list, akkademia, on=['sign', 'unicode'], how='outer')

        self.latin_to_uni = dict(zip(
            merged['sign'].astype(str),
            merged['unicode'].astype(str)
        ))
        self._build_reverse_map()
        self.loaded = True
        print(f"Loaded {len(self.latin_to_uni)} sign mappings from GitHub.")
        return self

    def load_manual_corrections(self, *csv_paths):
        """
        Load manual correction CSV files.
        Expects CSVs with columns mapping unmatched signs to Unicode.

        Args:
            csv_paths: Paths to correction CSV files.
        """
        import pandas as pd

        for path in csv_paths:
            try:
                df = pd.read_csv(path)
                if 'unmatched_sign' in df.columns and 'use' in df.columns:
                    corrections = df[['unmatched_sign', 'use']].dropna()
                    mapping = dict(zip(corrections['unmatched_sign'],
                                       corrections['use']))
                elif 'value' in df.columns and 'SIGN' in df.columns:
                    corrections = df[['value', 'SIGN']].dropna()
                    mapping = dict(zip(
                        corrections['value'].str.strip("[]' "),
                        corrections['SIGN']
                    ))
                else:
                    print(f"  Skipped {path}: unrecognized columns {df.columns.tolist()}")
                    continue

                self.manual_corrections.update(mapping)
                self.latin_to_uni.update(mapping)
                print(f"  Loaded {len(mapping)} corrections from {path}")
            except FileNotFoundError:
                print(f"  Not found: {path}")

        self._build_reverse_map()
        return self

    def _build_reverse_map(self):
        """Build Unicode → Latin reverse mapping."""
        self.uni_to_latin = {}
        for latin, uni in self.latin_to_uni.items():
            if uni and str(uni) != 'nan':
                self.uni_to_latin[uni] = latin

    def _normalize(self, text):
        """
        Normalize transliteration for sign lookup.
        Handles determinatives, hyphens, special chars.
        """
        if not text:
            return ''
        s = str(text)
        # Remove determinatives {d}, {m}, {f}, {ki}
        s = re.sub(r'\{[^}]*\}', '', s)
        # Hyphens and dots to spaces
        s = s.replace('-', ' ').replace('.', ' ')
        # Remove damage markers
        for ch in ['[', ']', '#', '!', '?', '*', '(', ')']:
            s = s.replace(ch, '')
        # Collapse whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _normalize_elamite(self, text):
        """
        Elamite-specific normalization (two-pass, matching original pipeline).
        """
        if not text:
            return ''
        s = str(text)
        s = re.sub(r"\(\s*md\s*\)", " m d ", s)
        s = re.sub(r"[.,:;!?()\[\]{}<>\\\"''/\\|*^`~]", " ", s)
        s = re.sub(r"[-\u2013\u2014]", " ", s)
        s = re.sub(r"([A-Za-z])(\d)([A-Za-z])", r"\1\2 \3", s)
        s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _lookup(self, sign):
        """Look up a single sign, trying multiple cases."""
        if sign in self.latin_to_uni:
            return self.latin_to_uni[sign]
        if sign.lower() in self.latin_to_uni:
            return self.latin_to_uni[sign.lower()]
        if sign.upper() in self.latin_to_uni:
            return self.latin_to_uni[sign.upper()]
        return None

    def to_unicode(self, text, lang=None):
        """
        Convert Latin transliteration to Unicode cuneiform.

        Args:
            text: Latin transliteration string (e.g. "a-na" or "a na").
            lang: Optional language for specialized normalization ('elx').

        Returns:
            Unicode cuneiform string (space-separated signs).
        """
        if not self.loaded:
            raise RuntimeError("Sign lists not loaded. Call load_sign_lists() first.")

        if lang == 'elx':
            normalized = self._normalize_elamite(text)
        else:
            normalized = self._normalize(text)

        if not normalized:
            return ''

        tokens = normalized.split()
        result = []
        for tok in tokens:
            uni = self._lookup(tok)
            if uni:
                result.append(uni)
            else:
                result.append(tok)  # keep as-is if not found

        return ' '.join(result)

    def to_unicode_detailed(self, text, lang=None):
        """
        Convert with detailed results including unmatched signs.

        Returns:
            Dict with 'unicode', 'clean' (bool), 'unmatched' (list).
        """
        if not self.loaded:
            raise RuntimeError("Sign lists not loaded. Call load_sign_lists() first.")

        if lang == 'elx':
            normalized = self._normalize_elamite(text)
        else:
            normalized = self._normalize(text)

        if not normalized:
            return {'unicode': '', 'clean': True, 'unmatched': []}

        tokens = normalized.split()
        result = []
        unmatched = []
        for tok in tokens:
            uni = self._lookup(tok)
            if uni:
                result.append(uni)
            else:
                result.append(tok)
                if re.search(r'[A-Za-z]', tok):
                    unmatched.append(tok)

        return {
            'unicode': ' '.join(result),
            'clean': len(unmatched) == 0,
            'unmatched': unmatched,
        }

    def to_unicode_batch(self, texts, lang=None):
        """Convert a list of transliterations to Unicode."""
        return [self.to_unicode(t, lang) for t in texts]

    def to_latin(self, unicode_text):
        """
        Convert Unicode cuneiform back to Latin sign names.

        Args:
            unicode_text: Unicode cuneiform string.

        Returns:
            Space-separated Latin sign names.
        """
        if not self.loaded:
            raise RuntimeError("Sign lists not loaded. Call load_sign_lists() first.")

        result = []
        for char in unicode_text:
            if char == ' ':
                continue
            if char in self.uni_to_latin:
                result.append(self.uni_to_latin[char])
            else:
                result.append(char)

        return ' '.join(result)

    def conversion_rate(self, texts, lang=None):
        """
        Compute conversion rate for a list of texts.

        Returns:
            Dict with 'total', 'clean', 'rate', 'unmatched_signs'.
        """
        total = len(texts)
        clean = 0
        all_unmatched = []

        for text in texts:
            result = self.to_unicode_detailed(text, lang)
            if result['clean']:
                clean += 1
            all_unmatched.extend(result['unmatched'])

        from collections import Counter
        return {
            'total': total,
            'clean': clean,
            'rate': clean / total if total > 0 else 0,
            'unmatched_signs': Counter(all_unmatched).most_common(20),
        }

    # =================================================================
    # EXPERT DICTIONARY UPDATES
    # =================================================================

    def add_sign(self, latin, unicode, source="manual"):
        """
        Add or update a single sign mapping.

        Args:
            latin: Latin sign name (e.g. "ŠU₂", "a", "KUR").
            unicode: Unicode cuneiform character(s).
            source: Attribution string (e.g. "Anderson 2026", "CDLI").
        """
        old = self.latin_to_uni.get(latin)
        self.latin_to_uni[latin] = unicode
        self.manual_corrections[latin] = unicode
        self._update_log.append({
            'action': 'update' if old else 'add',
            'latin': latin,
            'unicode': unicode,
            'previous': old,
            'source': source,
        })
        self._build_reverse_map()

    def add_signs(self, mappings, source="manual"):
        """
        Add multiple sign mappings at once.

        Args:
            mappings: Dict of {latin: unicode} pairs,
                      or list of (latin, unicode) tuples.
            source: Attribution string.
        """
        if isinstance(mappings, dict):
            mappings = mappings.items()
        for latin, unicode in mappings:
            self.add_sign(latin, unicode, source)

    def update_from_file(self, filepath, source=None):
        """
        Load expert corrections from a file. Auto-detects format.

        Supported formats:
            .json   — {"sign": "unicode", ...} or [{"latin": "x", "unicode": "y"}, ...]
            .csv    — Columns: latin,unicode (or sign,unicode or reading,cuneiform)
            .tsv    — Same as CSV but tab-delimited
            .txt    — One pair per line: "latin<tab>unicode" or "latin,unicode"
            .xlsx   — First two columns treated as latin, unicode

        Args:
            filepath: Path to corrections file.
            source: Attribution (default: filename).

        Returns:
            Number of mappings added.
        """
        path = Path(filepath)
        if source is None:
            source = path.name

        ext = path.suffix.lower()
        count = 0

        if ext == '.json':
            count = self._load_json_updates(path, source)
        elif ext == '.csv':
            count = self._load_delimited_updates(path, ',', source)
        elif ext == '.tsv':
            count = self._load_delimited_updates(path, '\t', source)
        elif ext == '.txt':
            count = self._load_txt_updates(path, source)
        elif ext in ('.xlsx', '.xls'):
            count = self._load_excel_updates(path, source)
        else:
            raise ValueError(
                f"Unsupported format: {ext}. "
                f"Use .json, .csv, .tsv, .txt, or .xlsx"
            )

        print(f"  Loaded {count} mappings from {path.name} (source: {source})")
        return count

    def _load_json_updates(self, path, source):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        count = 0
        if isinstance(data, dict):
            for latin, unicode in data.items():
                if latin and unicode:
                    self.add_sign(str(latin).strip(), str(unicode).strip(), source)
                    count += 1
        elif isinstance(data, list):
            for entry in data:
                latin = entry.get('latin') or entry.get('sign') or entry.get('reading')
                unicode = entry.get('unicode') or entry.get('cuneiform')
                if latin and unicode:
                    self.add_sign(str(latin).strip(), str(unicode).strip(), source)
                    count += 1
        return count

    def _load_delimited_updates(self, path, delimiter, source):
        import csv
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)

            # Detect column names
            if header:
                header_lower = [h.strip().lower() for h in header]
                latin_col, uni_col = None, None

                for i, h in enumerate(header_lower):
                    if h in ('latin', 'sign', 'reading', 'transliteration',
                             'unmatched_sign', 'value', 'name'):
                        latin_col = i
                    elif h in ('unicode', 'cuneiform', 'use', 'sign_unicode',
                               'resolved'):
                        uni_col = i

                # Fall back to first two columns
                if latin_col is None:
                    latin_col = 0
                if uni_col is None:
                    uni_col = 1 if latin_col == 0 else 0

                for row in reader:
                    if len(row) > max(latin_col, uni_col):
                        latin = row[latin_col].strip()
                        unicode = row[uni_col].strip()
                        if latin and unicode:
                            self.add_sign(latin, unicode, source)
                            count += 1
        return count

    def _load_txt_updates(self, path, source):
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Try tab first, then comma
                if '\t' in line:
                    parts = line.split('\t', 1)
                elif ',' in line:
                    parts = line.split(',', 1)
                else:
                    continue
                if len(parts) == 2:
                    latin, unicode = parts[0].strip(), parts[1].strip()
                    if latin and unicode:
                        self.add_sign(latin, unicode, source)
                        count += 1
        return count

    def _load_excel_updates(self, path, source):
        import pandas as pd
        df = pd.read_excel(path)
        count = 0

        # Detect columns
        cols = [c.strip().lower() for c in df.columns]
        latin_col, uni_col = None, None

        for i, c in enumerate(cols):
            if c in ('latin', 'sign', 'reading', 'transliteration', 'name'):
                latin_col = df.columns[i]
            elif c in ('unicode', 'cuneiform', 'resolved', 'use'):
                uni_col = df.columns[i]

        if latin_col is None:
            latin_col = df.columns[0]
        if uni_col is None:
            uni_col = df.columns[1]

        for _, row in df.iterrows():
            latin = str(row[latin_col]).strip()
            unicode = str(row[uni_col]).strip()
            if latin and unicode and latin != 'nan' and unicode != 'nan':
                self.add_sign(latin, unicode, source)
                count += 1
        return count

    def export_unmatched(self, texts, filepath, lang=None):
        """
        Run conversion on texts and export unmatched signs as a template
        for experts to fill in.

        Creates a CSV with columns: sign, unicode, source
        The 'unicode' column is blank — experts fill it in, then reload
        with update_from_file().

        Args:
            texts: List of transliteration strings to check.
            filepath: Output CSV path.
            lang: Optional language code.
        """
        from collections import Counter
        all_unmatched = Counter()

        for text in texts:
            result = self.to_unicode_detailed(text, lang)
            for sign in result['unmatched']:
                all_unmatched[sign] += 1

        path = Path(filepath)
        with open(path, 'w', encoding='utf-8') as f:
            f.write("sign,unicode,count,source\n")
            for sign, count in all_unmatched.most_common():
                f.write(f"{sign},,{count},\n")

        print(f"  Exported {len(all_unmatched)} unmatched signs to {path.name}")
        print(f"  Fill in the 'unicode' column and reload with update_from_file()")

    def generate_template(self, filepath, n_empty=20):
        """
        Generate a blank template file for experts to add new mappings.

        Args:
            filepath: Output path (.csv, .tsv, .json, or .txt).
            n_empty: Number of empty rows to include.
        """
        path = Path(filepath)
        ext = path.suffix.lower()

        if ext == '.csv':
            with open(path, 'w', encoding='utf-8') as f:
                f.write("sign,unicode,source\n")
                f.write("# Example: ŠU₂,𒋗,Anderson 2026\n")
                for _ in range(n_empty):
                    f.write(",,\n")
        elif ext == '.tsv':
            with open(path, 'w', encoding='utf-8') as f:
                f.write("sign\tunicode\tsource\n")
                f.write("# Example: ŠU₂\t𒋗\tAnderson 2026\n")
                for _ in range(n_empty):
                    f.write("\t\t\n")
        elif ext == '.json':
            template = {
                "_instructions": "Add sign mappings below. Keys are Latin readings, values are Unicode cuneiform.",
                "_example": {"ŠU₂": "𒋗"},
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(template, f, ensure_ascii=False, indent=2)
        elif ext == '.txt':
            with open(path, 'w', encoding='utf-8') as f:
                f.write("# cunei-conv sign corrections\n")
                f.write("# Format: sign<TAB>unicode\n")
                f.write("# Example:\n")
                f.write("# ŠU₂\t𒋗\n")
                f.write("\n")
        else:
            raise ValueError(f"Unsupported template format: {ext}")

        print(f"  Template created: {path.name}")

    def update_log(self):
        """Print the history of manual additions and corrections."""
        if not self._update_log:
            print("No updates recorded.")
            return
        print(f"Update history ({len(self._update_log)} entries):")
        for entry in self._update_log:
            action = entry['action']
            if action == 'update':
                print(f"  UPDATED {entry['latin']}: {entry['previous']} → "
                      f"{entry['unicode']} (source: {entry['source']})")
            else:
                print(f"  ADDED   {entry['latin']} → {entry['unicode']} "
                      f"(source: {entry['source']})")

    # =================================================================
    # SAVE / LOAD / STATS
    # =================================================================

    def save(self, filepath):
        """Save sign dictionary to JSON, including update history."""
        data = {
            'latin_to_uni': self.latin_to_uni,
            'manual_corrections': self.manual_corrections,
            'update_log': self._update_log,
            'version': '0.1.0',
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath):
        """Load sign dictionary from JSON."""
        conv = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        conv.latin_to_uni = data['latin_to_uni']
        conv.manual_corrections = data.get('manual_corrections', {})
        conv._update_log = data.get('update_log', [])
        conv._build_reverse_map()
        conv.loaded = True
        return conv

    def stats(self):
        """Print converter statistics."""
        if not self.loaded:
            print("Not loaded.")
            return
        print(f"Sign mappings:       {len(self.latin_to_uni):,}")
        print(f"Reverse mappings:    {len(self.uni_to_latin):,}")
        print(f"Manual corrections:  {len(self.manual_corrections):,}")
        print(f"Update log entries:  {len(self._update_log):,}")
