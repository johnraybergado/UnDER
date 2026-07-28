from .tmp_dataset import TMPDataset
from .kuas_train_dataset import KUASTrainDataset
from .ug_train_dataset import UGTrainDataset

__datasets__ = {
    "tmp": TMPDataset,
    "kuas_train": KUASTrainDataset,
    "ug_train": UGTrainDataset
}
