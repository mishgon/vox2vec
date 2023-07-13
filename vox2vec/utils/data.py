import random

from torch.utils.data import Dataset

import pdp


class VanillaDataset(Dataset):
    def __init__(self, ids, load_func):
        self.ids = ids
        self.load_func = load_func

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.load_func(self.ids[i])


class Pool(Dataset):
    def __init__(
            self,
            dataset: Dataset,
            num_samples: int,
            num_workers: int,
            buffer_size: int
    ) -> None:
        """Wrapper of ``dataset`` (torch.utils.data.Dataset),
        that creates a buffer in RAM and ``num_workers`` processes
        that constantly fill the buffer with random examples from ``dataset``.
        This allows to load new examples in parallel with the following pipeline steps,
        e.g., forward-backward pass, etc.

        Args:
            dataset (Dataset): original dataset to wrap.
            num_samples (int): size of the resulting dataset, can be arbitary number. 
            num_workers (int): number of sampling processes.
            buffer_size (int): maximal number of examples in the buffer.
        """
        self.num_samples = num_samples

        def src():
            while True:
                yield random.randint(0, len(dataset) - 1)

        self.pipeline = pdp.Pipeline(
            pdp.Source(src(), buffer_size=1),
            pdp.One2One(dataset.__getitem__, n_workers=num_workers, buffer_size=buffer_size),
        )
        self.pool = iter(self.pipeline)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        return next(self.pool)

    def __del__(self):
        self.pipeline.close()
