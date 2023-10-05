import torch

from huggingface_hub import hf_hub_download, scan_cache_dir

from .fpn import FPN3d


def vox2vec_contrastive() -> FPN3d:
    model = FPN3d(in_channels=1, base_channels=16, num_scales=6)

    weights_path = hf_hub_download(
        repo_id='mishgon/vox2vec',
        filename='vox2vec_contrastive.pt',
        revision='29dfa9718434a58b3241af3c43dca90c867bea15'
    )
    model.load_state_dict(torch.load(weights_path))

    # print('Removing cache 🗑️ ...')
    # revision = weights_path.split('/')[-4]
    # delete_strategy = scan_cache_dir().delete_revisions(revision)
    # delete_strategy.execute()

    return model


def vox2vec_vicreg() -> FPN3d:
    model = FPN3d(in_channels=1, base_channels=16, num_scales=6)

    weights_path = hf_hub_download(
        repo_id='mishgon/vox2vec',
        filename='vox2vec_vicreg.pt',
        revision='29dfa9718434a58b3241af3c43dca90c867bea15'
    )
    model.load_state_dict(torch.load(weights_path))

    return model
