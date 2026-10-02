"""Release dataset-card frontmatter tests (spec §6, §12).

The Hub validates README.md frontmatter on every commit: an unknown or wrongly
cased license id rejects the WHOLE upload ("Invalid metadata in README.md").
These tests pin the ids we emit to the Hub's known set.
"""

import re

from ornix_dataset.exporters.card import render_dataset_card

_CARD = re.compile(r"^---\n(.*?)\n---\n", re.S)


def _frontmatter(rights):
    rows = [{"language": "vi", "split": "train", "duration_s": 1.0}]
    card = render_dataset_card("rel-1", rows, rights, {})
    m = _CARD.match(card)
    assert m, card[:200]
    return m.group(1)


def _licenses(rights):
    fm = _frontmatter(rights)
    out, in_block = [], False
    for line in fm.splitlines():
        if line.startswith("license:"):
            in_block = True
            continue
        if in_block:
            if line.startswith("  - "):
                out.append(line[4:].strip())
            else:
                break
    return out


def test_known_license_lowercased():
    assert _licenses({"licenses": ["CC-BY-NC-4.0"]}) == ["cc-by-nc-4.0"]


def test_mit_and_mixed():
    assert _licenses({"licenses": ["MIT", "CC-BY-4.0"]}) == ["cc-by-4.0", "mit"]


def test_operator_declared_maps_to_other():
    # "UNSPECIFIED-OPERATOR-DECLARED" is not a Hub id; it must become "other"
    assert _licenses(
        {"licenses": ["UNSPECIFIED-OPERATOR-DECLARED"]}) == ["other"]


def test_missing_licenses_never_emits_placeholder():
    # "see-source-terms" used to be emitted and is NOT a valid Hub license id
    assert _licenses({}) == ["other"]
    assert "see-source-terms" not in _frontmatter({})


def test_unknown_sentinel_maps_to_other():
    assert _licenses({"licenses": ["UNKNOWN"]}) == ["other"]


def test_frontmatter_is_valid_yaml():
    import yaml

    rows = [{"language": "vi", "split": "train", "duration_s": 1.0}]
    card = render_dataset_card("r", rows, {"licenses": ["CC-BY-NC-4.0"]}, {})
    fm = yaml.safe_load(_CARD.match(card).group(1))
    assert fm["license"] == ["cc-by-nc-4.0"]
    assert "text-to-speech" in fm["task_categories"]