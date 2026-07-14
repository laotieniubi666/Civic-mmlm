# Real-data acquisition and adapter contract

The repository intentionally does not redistribute external datasets or silently download license-gated resources. A strict experiment should create a manifest for every source.

## Paper-listed resources

| Resource | Suggested official source | Local role |
|---|---|---|
| MultiEURLEX | `https://huggingface.co/datasets/nlpaueb/multi_eurlex` and the authors' repository | multilingual policy routing |
| TyDi QA | `https://github.com/google-research-datasets/tydiqa` | evidence-bounded multilingual QA |
| BUFFET | official repository linked by the paper/authors | few-shot cross-lingual transfer |
| M5 | official repository linked by the paper/authors | multilingual and multicultural VQA |
| MiLiC-Eval | official repository linked by the paper/authors | minority-language evaluation |
| xBD / xView2 | `https://xview2.org/dataset` | pre/post-disaster damage assessment |
| World Development Indicators | `https://api.worldbank.org/v2/` or WDI bulk files | socioeconomic indicators |
| geoBoundaries | `https://www.geoboundaries.org/` | administrative boundaries |

Always verify current license terms at the source. Some resources may require registration, an access agreement, or task-specific citation.

## Required manifest

Create one YAML record per snapshot:

```yaml
name: multi_eurlex
source: https://huggingface.co/datasets/nlpaueb/multi_eurlex
revision: <commit-or-tag>
downloaded_at_utc: 2026-07-14T00:00:00Z
files:
  - path: raw/train.parquet
    sha256: <hash>
license: <verified-license>
exclusions:
  - <rule>
transformations:
  - normalize Unicode without translating the target-language text
split_unit: document_id
split_manifest_sha256: <hash>
```

## Adapter output

Every task adapter must convert raw records to the same evidence contract:

```python
sample = {
    "sample_id": str,
    "modalities": {
        "text": {
            "atoms": FloatTensor[N_text, D_raw],
            "mask": BoolTensor[N_text],
            "provenance": FloatTensor[N_text],
            "reliability": FloatTensor[N_text],
            "contradiction": FloatTensor[N_text],
        },
        "image": {...},
        "geo": {...},
        "table": {...},
    },
    "valid_modalities": {...},
    "material_modalities": {...},
    "label": LongTensor[],
    "group": LongTensor[],
    "need_bin": LongTensor[],
    "legality": FloatTensor[A],
    "budget": FloatTensor[],
    "action_costs": FloatTensor[A],
    "identification": LongTensor[],
}
```

## Provenance and contradiction fields

`provenance` should measure traceability, not model confidence. Examples include an official clause with stable document identifiers, an image with acquisition metadata, or a table row linked to a published indicator code.

`reliability` should represent source/acquisition quality, such as image cloud cover, OCR quality, sensor validity, translation audit status, or missing-value imputation confidence.

`contradiction` should be computed before action selection where possible. A practical implementation can combine cross-encoder entailment, rule-based policy conflicts, temporal inconsistency, and geospatial mismatch. Store the method and threshold in the manifest.

## Split discipline

- MultiEURLEX: preserve chronological or document-source separation where the chosen release supports it.
- QA: prevent question/passage duplicates and translated variants from crossing splits.
- xBD: split by disaster event or geographic unit, not random image tiles alone.
- Geospatial allocation: split by geographic unit and time so nearby duplicates do not leak.
- Calibration: the conformal calibration set must be disjoint from training and final testing.

## Why automatic download is not the default

Several datasets are large, versioned independently, or governed by separate license terms. An automatic downloader would make it too easy to lose the exact snapshot or accept terms implicitly. The included `prepare_data.py` therefore prints the registry and required actions rather than pretending that a single command can create a defensible benchmark.
