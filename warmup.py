from argparse import ArgumentParser
from tqdm import tqdm

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from vox2vec.default_params import *
from vox2vec.pretrain.data import PretrainDataset
from vox2vec.utils.data import VanillaDataset


def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('--cache_dir', required=True)

    parser.add_argument('--amos_dir')
    parser.add_argument('--flare_dir')
    parser.add_argument('--nlst_dir')
    parser.add_argument('--lits_dir')
    parser.add_argument('--midrc_dir')
    parser.add_argument('--nsclc_dir')

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=10)

    parser.add_argument('--spacing', nargs='+', type=float, default=SPACING)
    parser.add_argument('--patch_size', nargs='+', type=int, default=PATCH_SIZE)

    return parser.parse_args()


def main(args):
    dm = PretrainDataset(
        cache_dir=args.cache_dir,
        spacing=tuple(args.spacing),
        patch_size=tuple(args.patch_size),
        window_hu=WINDOW_HU,
        min_window_hu=MIN_WINDOW_HU,
        max_window_hu=MAX_WINDOW_HU,
        max_num_voxels_per_patch=MAX_NUM_VOXELS_PER_PATCH,
        batch_size=None,
        amos_dir=args.amos_dir,
        flare_dir=args.flare_dir,
        nlst_dir=args.nlst_dir,
        midrc_dir=args.midrc_dir,
        nsclc_dir=args.nsclc_dir,
    )
    ds = VanillaDataset(dm.ids, dm.load_example)
    dl = DataLoader(ds, args.batch_size, num_workers=args.num_workers, collate_fn=lambda x: x)

    for batch in tqdm(dl, 'Warming up ChestAbdominalPublic'):
        pass


if __name__ == '__main__':
    main(parse_args())
