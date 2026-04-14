"""
Tests for cunei-tools package.

Run with: python -m pytest tests/ -v
"""

import json
import os
import tempfile
import pytest

from cunei_tools.segmenter import CuneiSeg, LANG_THRESHOLDS
from cunei_tools.converter import CuneiConv


# ============================================================
# Test Data
# ============================================================

# Simulated Unicode cuneiform documents (using real Unicode cuneiform chars)
SAMPLE_DOCS = [
    "𒀀𒈾 𒁀𒌅 𒊭",
    "𒂍 𒈾 𒀀𒈾 𒁀𒌅",
    "𒊭 𒀀𒈾 𒂍",
    "𒁀𒌅 𒊭 𒂍 𒈾",
    "𒀀𒈾 𒂍 𒊭 𒁀𒌅 𒈾",
]

SAMPLE_SIGN_DICT = {
    'a': '𒀀',
    'na': '𒈾',
    'ba': '𒁀',
    'ab': '𒀊',
    'an': '𒀭',
    'ki': '𒆠',
    'en': '𒂗',
    'šu': '𒋗',
    'lu': '𒇽',
    'gal': '𒃲',
    'dingir': '𒀭',
}


# ============================================================
# CuneiSeg Tests
# ============================================================

class TestCuneiSegInit:
    """Test segmenter initialization."""

    def test_default_threshold(self):
        seg = CuneiSeg()
        assert seg.threshold == 0.65
        assert seg.lang is None
        assert seg.trained is False

    def test_lang_threshold(self):
        for lang, expected in LANG_THRESHOLDS.items():
            seg = CuneiSeg(lang=lang)
            assert seg.threshold == expected
            assert seg.lang == lang

    def test_custom_threshold(self):
        seg = CuneiSeg(threshold=0.42)
        assert seg.threshold == 0.42

    def test_custom_overrides_lang(self):
        seg = CuneiSeg(lang='akk', threshold=0.50)
        assert seg.threshold == 0.50


class TestCuneiSegTraining:
    """Test segmenter training."""

    def test_train_basic(self):
        seg = CuneiSeg()
        seg.train(SAMPLE_DOCS)
        assert seg.trained is True
        assert len(seg.unigrams) > 0
        assert len(seg.bigrams) > 0

    def test_train_counts(self):
        seg = CuneiSeg()
        seg.train(["𒀀𒈾 𒁀𒌅"])
        assert seg.unigrams['𒀀'] == 1
        assert seg.unigrams['𒈾'] == 1
        assert seg.unigrams['𒁀'] == 1
        assert seg.unigrams['𒌅'] == 1
        assert seg.bigrams[('𒀀', '𒈾')] == 1
        assert seg.bigrams[('𒁀', '𒌅')] == 1
        # Cross-boundary pair IS counted (spaces stripped before bigram counting)
        # This is correct: these bigrams get low TP scores, enabling boundary detection
        assert seg.bigrams[('𒈾', '𒁀')] == 1

    def test_train_empty_docs(self):
        seg = CuneiSeg()
        seg.train(["", "  ", None])
        assert seg.trained is True
        assert len(seg.unigrams) == 0

    def test_train_from_file(self):
        seg = CuneiSeg()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False, encoding='utf-8') as f:
            for doc in SAMPLE_DOCS:
                f.write(doc + '\n')
            path = f.name
        try:
            seg.train_from_file(path)
            assert seg.trained is True
            assert len(seg.unigrams) > 0
        finally:
            os.unlink(path)


class TestCuneiSegSegmentation:
    """Test segmenter output."""

    def setup_method(self):
        self.seg = CuneiSeg(threshold=0.5)
        self.seg.train(SAMPLE_DOCS)

    def test_segment_untrained(self):
        seg = CuneiSeg()
        with pytest.raises(RuntimeError, match="not trained"):
            seg.segment("𒀀𒈾𒁀𒌅")

    def test_segment_empty(self):
        assert self.seg.segment("") == ""
        assert self.seg.segment("𒀀") == "𒀀"

    def test_segment_returns_string(self):
        result = self.seg.segment("𒀀𒈾𒁀𒌅")
        assert isinstance(result, str)

    def test_segment_has_spaces(self):
        result = self.seg.segment("𒀀𒈾𒁀𒌅𒊭")
        assert ' ' in result

    def test_segment_batch(self):
        results = self.seg.segment_batch(["𒀀𒈾𒁀𒌅", "𒊭𒂍"])
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_threshold_override(self):
        result_low = self.seg.segment("𒀀𒈾𒁀𒌅", threshold=0.1)
        result_high = self.seg.segment("𒀀𒈾𒁀𒌅", threshold=0.99)
        # Lower threshold = fewer boundaries, higher = more
        assert result_low.count(' ') <= result_high.count(' ')


class TestCuneiSegOptimize:
    """Test threshold optimization."""

    def setup_method(self):
        self.seg = CuneiSeg()
        self.seg.train(SAMPLE_DOCS)

    def test_find_optimal_threshold(self):
        metrics = self.seg.find_optimal_threshold(SAMPLE_DOCS)
        assert 'threshold' in metrics
        assert 'f1' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 0 <= metrics['f1'] <= 1
        assert 0 < metrics['threshold'] < 1


class TestCuneiSegPersistence:
    """Test save/load."""

    def test_save_load_roundtrip(self):
        seg = CuneiSeg(lang='akk')
        seg.train(SAMPLE_DOCS)
        result_before = seg.segment("𒀀𒈾𒁀𒌅")

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            seg.save(path)
            seg2 = CuneiSeg.load(path)
            assert seg2.lang == 'akk'
            assert seg2.threshold == seg.threshold
            assert seg2.trained is True
            assert seg2.segment("𒀀𒈾𒁀𒌅") == result_before
        finally:
            os.unlink(path)

    def test_save_format(self):
        seg = CuneiSeg(lang='sux')
        seg.train(SAMPLE_DOCS)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            seg.save(path)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert 'lang' in data
            assert 'threshold' in data
            assert 'unigrams' in data
            assert 'bigrams' in data
            assert 'version' in data
        finally:
            os.unlink(path)


# ============================================================
# CuneiConv Tests
# ============================================================

def make_converter():
    """Create a converter with sample sign dictionary."""
    conv = CuneiConv()
    conv.latin_to_uni = SAMPLE_SIGN_DICT.copy()
    conv._build_reverse_map()
    conv.loaded = True
    return conv


class TestCuneiConvInit:
    """Test converter initialization."""

    def test_default_state(self):
        conv = CuneiConv()
        assert conv.loaded is False
        assert len(conv.latin_to_uni) == 0

    def test_not_loaded_raises(self):
        conv = CuneiConv()
        with pytest.raises(RuntimeError, match="not loaded"):
            conv.to_unicode("a-na")


class TestCuneiConvConversion:
    """Test conversion functions."""

    def setup_method(self):
        self.conv = make_converter()

    def test_to_unicode_basic(self):
        result = self.conv.to_unicode("a-na")
        assert '𒀀' in result
        assert '𒈾' in result

    def test_to_unicode_dots(self):
        result = self.conv.to_unicode("a.na")
        assert '𒀀' in result
        assert '𒈾' in result

    def test_to_unicode_unmatched(self):
        result = self.conv.to_unicode("a-na foo")
        assert 'foo' in result

    def test_to_unicode_determinative(self):
        result = self.conv.to_unicode("{d}en-ki")
        assert '𒂗' in result
        assert '𒆠' in result
        assert '{d}' not in result

    def test_to_unicode_empty(self):
        assert self.conv.to_unicode("") == ''
        assert self.conv.to_unicode(None) == ''

    def test_to_unicode_batch(self):
        results = self.conv.to_unicode_batch(["a-na", "ba-ab"])
        assert len(results) == 2

    def test_to_unicode_detailed(self):
        result = self.conv.to_unicode_detailed("a-na")
        assert result['clean'] is True
        assert result['unmatched'] == []
        assert '𒀀' in result['unicode']

    def test_to_unicode_detailed_unmatched(self):
        result = self.conv.to_unicode_detailed("a-na foo-bar")
        assert result['clean'] is False
        assert 'foo' in result['unmatched']
        assert 'bar' in result['unmatched']

    def test_to_latin(self):
        result = self.conv.to_latin("𒀀𒈾")
        assert 'a' in result
        assert 'na' in result

    def test_case_insensitive_lookup(self):
        self.conv.latin_to_uni['KUR'] = '𒆳'
        assert '𒆳' in self.conv.to_unicode("kur")
        assert '𒆳' in self.conv.to_unicode("KUR")

    def test_conversion_rate(self):
        stats = self.conv.conversion_rate(["a-na", "ba-ab", "foo-bar"])
        assert stats['total'] == 3
        assert stats['clean'] == 2
        assert abs(stats['rate'] - 0.667) < 0.01


class TestCuneiConvExpertUpdates:
    """Test expert dictionary update features."""

    def setup_method(self):
        self.conv = make_converter()

    def test_add_sign(self):
        self.conv.add_sign('kur', '𒆳', source='test')
        assert self.conv.latin_to_uni['kur'] == '𒆳'
        assert self.conv.manual_corrections['kur'] == '𒆳'
        assert '𒆳' in self.conv.to_unicode("a-na kur")

    def test_add_signs_dict(self):
        self.conv.add_signs({'kur': '𒆳', 'uru': '𒌷'}, source='batch')
        assert 'kur' in self.conv.latin_to_uni
        assert 'uru' in self.conv.latin_to_uni

    def test_add_signs_list(self):
        self.conv.add_signs([('kur', '𒆳'), ('uru', '𒌷')], source='batch')
        assert 'kur' in self.conv.latin_to_uni
        assert 'uru' in self.conv.latin_to_uni

    def test_update_overwrites(self):
        self.conv.add_sign('a', '𒀀', source='original')
        self.conv.add_sign('a', '𒁁', source='correction')
        assert self.conv.latin_to_uni['a'] == '𒁁'
        assert len(self.conv._update_log) == 2
        assert self.conv._update_log[1]['action'] == 'update'

    def test_update_from_csv(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False, encoding='utf-8') as f:
            f.write("sign,unicode\n")
            f.write("kur,𒆳\n")
            f.write("uru,𒌷\n")
            path = f.name
        try:
            count = self.conv.update_from_file(path, source='test_csv')
            assert count == 2
            assert 'kur' in self.conv.latin_to_uni
        finally:
            os.unlink(path)

    def test_update_from_tsv(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv',
                                         delete=False, encoding='utf-8') as f:
            f.write("reading\tunicode\n")
            f.write("kur\t𒆳\n")
            path = f.name
        try:
            count = self.conv.update_from_file(path)
            assert count == 1
        finally:
            os.unlink(path)

    def test_update_from_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False, encoding='utf-8') as f:
            json.dump({'kur': '𒆳', 'uru': '𒌷'}, f)
            path = f.name
        try:
            count = self.conv.update_from_file(path, source='test_json')
            assert count == 2
        finally:
            os.unlink(path)

    def test_update_from_txt(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False, encoding='utf-8') as f:
            f.write("# comment\n")
            f.write("kur\t𒆳\n")
            f.write("uru,𒌷\n")
            path = f.name
        try:
            count = self.conv.update_from_file(path)
            assert count == 2
        finally:
            os.unlink(path)

    def test_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                self.conv.update_from_file(path)
        finally:
            os.unlink(path)

    def test_export_unmatched(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            self.conv.export_unmatched(["a-na foo bar", "baz"], path)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert 'foo' in content
            assert 'bar' in content
            assert 'baz' in content
            assert 'sign,unicode,count,source' in content
        finally:
            os.unlink(path)

    def test_generate_template_csv(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            self.conv.generate_template(path, n_empty=5)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            assert 'sign,unicode,source' in lines[0]
        finally:
            os.unlink(path)

    def test_generate_template_json(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            self.conv.generate_template(path)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert '_instructions' in data
        finally:
            os.unlink(path)

    def test_update_log(self):
        self.conv.add_sign('kur', '𒆳', source='Anderson')
        self.conv.add_sign('uru', '𒌷', source='Thompson')
        assert len(self.conv._update_log) == 2
        assert self.conv._update_log[0]['source'] == 'Anderson'
        assert self.conv._update_log[1]['source'] == 'Thompson'


class TestCuneiConvPersistence:
    """Test save/load with updates preserved."""

    def test_save_load_roundtrip(self):
        conv = make_converter()
        conv.add_sign('kur', '𒆳', source='test')

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            conv.save(path)
            conv2 = CuneiConv.load_from_file(path)
            assert conv2.loaded is True
            assert 'kur' in conv2.latin_to_uni
            assert conv2.latin_to_uni['kur'] == '𒆳'
            assert len(conv2._update_log) == 1
            assert conv2.to_unicode("a-na kur") == conv.to_unicode("a-na kur")
        finally:
            os.unlink(path)

    def test_reverse_map_preserved(self):
        conv = make_converter()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            conv.save(path)
            conv2 = CuneiConv.load_from_file(path)
            assert conv2.to_latin("𒀀𒈾") == conv.to_latin("𒀀𒈾")
        finally:
            os.unlink(path)


class TestCuneiConvElamite:
    """Test Elamite-specific normalization."""

    def setup_method(self):
        self.conv = make_converter()

    def test_elamite_normalization(self):
        result = self.conv.to_unicode("a-na", lang='elx')
        assert '𒀀' in result

    def test_elamite_md_handling(self):
        # Elamite normalization expands (md) to m d
        result = self.conv.to_unicode("(md)a-na", lang='elx')
        assert isinstance(result, str)


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """End-to-end workflow tests."""

    def test_convert_then_segment(self):
        """Full pipeline: transliteration → Unicode → segmentation."""
        conv = make_converter()
        seg = CuneiSeg(threshold=0.5)

        # Convert documents
        latin_docs = ["a-na ba-ab", "en-ki an-na", "lu-gal a-na"]
        unicode_docs = [conv.to_unicode(d) for d in latin_docs]

        # Train segmenter on converted docs
        seg.train(unicode_docs)
        assert seg.trained is True

        # Segment unsegmented text
        unsegmented = unicode_docs[0].replace(' ', '')
        result = seg.segment(unsegmented)
        assert isinstance(result, str)
        assert len(result) > len(unsegmented)  # spaces added

    def test_expert_workflow(self):
        """Expert correction workflow: export → fix → reload."""
        conv = make_converter()

        # Find unmatched signs
        texts = ["a-na kur-ra", "uru-ki"]
        result = conv.to_unicode_detailed("a-na kur-ra")
        assert not result['clean']  # 'kur' and 'ra' unmatched

        # Export for expert
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False,
                                         mode='w', encoding='utf-8') as f:
            path = f.name
        try:
            conv.export_unmatched(texts, path)

            # Expert fills in corrections
            with open(path, 'w', encoding='utf-8') as f:
                f.write("sign,unicode,count,source\n")
                f.write("kur,𒆳,2,expert\n")
                f.write("ra,𒊏,1,expert\n")
                f.write("uru,𒌷,1,expert\n")

            # Reload corrections
            conv.update_from_file(path, source='expert review')
            result2 = conv.to_unicode_detailed("a-na kur-ra")
            assert result2['clean'] is True
        finally:
            os.unlink(path)
