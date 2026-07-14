from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    task: str
    access: str
    notes: str


DATASET_REGISTRY: Dict[str, DatasetRecord] = {
    "multi_eurlex": DatasetRecord(
        name="MultiEURLEX",
        task="multilingual multi-label policy classification",
        access="Hugging Face dataset: nlpaueb/multi_eurlex",
        notes="Use an explicit revision and preserve the chronological split metadata.",
    ),
    "tydiqa": DatasetRecord(
        name="TyDi QA",
        task="multilingual evidence-bounded question answering",
        access="Official Google Research dataset repository",
        notes="The public reference package does not redistribute the dataset.",
    ),
    "buffet": DatasetRecord(
        name="BUFFET",
        task="few-shot cross-lingual transfer",
        access="Authors' official release or paper repository",
        notes="Pin the exact task subset and prompt template used in an experiment.",
    ),
    "m5": DatasetRecord(
        name="M5",
        task="multilingual and multicultural vision-language evaluation",
        access="Authors' official release",
        notes="Record image licenses and task-specific evaluation scripts.",
    ),
    "milic_eval": DatasetRecord(
        name="MiLiC-Eval",
        task="minority-language evaluation",
        access="Authors' official release",
        notes="Do not translate away the target language in the native-language condition.",
    ),
    "xbd": DatasetRecord(
        name="xBD / xView2",
        task="pre/post-disaster damage assessment",
        access="Official xView2 dataset portal",
        notes="Registration or acceptance of dataset terms may be required.",
    ),
    "wdi": DatasetRecord(
        name="World Development Indicators",
        task="socioeconomic indicators",
        access="World Bank Indicators API or bulk download",
        notes="Store indicator codes, retrieval date, and source metadata.",
    ),
    "geoboundaries": DatasetRecord(
        name="geoBoundaries",
        task="administrative boundaries and geospatial features",
        access="Official geoBoundaries release",
        notes="Pin release version, boundary type, and license.",
    ),
}
