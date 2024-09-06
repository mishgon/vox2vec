from dataclasses import dataclass
from typing import Union


@dataclass
class SourceDataDirs:
    nlst: str
    amos: str
    abdomen_atlas: str
    flare23: str
    lidc: str
    midrc_ricord_1a: str


@dataclass
class PreparedDataDirs:
    nlst: str
    amos_ct_labeled_train: str
    amos_ct_unlabeled_train: str
    amos_ct_val: str
    abdomen_atlas: str
    flare23_labeled_train: str
    flare23_unlabeled_train: str
    flare23_labeled_val: str
    lidc: str
    midrc_ricord_1a: str


@dataclass
class PretrainDataFractions:
    nlst: Union[float, int] = 1.0
    amos_ct_labeled_train: Union[float, int] = 1.0
    amos_ct_unlabeled_train: Union[float, int] = 1.0
    abdomen_atlas: Union[float, int] = 1.0
    flare23_labeled_train: Union[float, int] = 1.0
    flare23_unlabeled_train: Union[float, int] = 1.0
