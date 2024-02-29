import torch
from ..model import VQMDVAE
from ..data import MeadDataset


def analysis_resynthesis(model: VQMDVAE,
                         dataset: MeadDataset,
                         information: dict = None,
                         seq_length: int = 60,
                         path_to_save: str = None):
    a, v = dataset.read(**information)
    assert a.shape[0] > seq_length, print("Problem with sequence length !")
    a, v = torch.from_numpy(a[:seq_length]).unsqueeze(0).to("cuda"), torch.from_numpy(v[:seq_length]).unsqueeze(0).to("cuda")
    _, a_rec, v_rec = model(a, v)
    if path_to_save is not None:
        model.visual_reconstruction(v_rec[0], add=f"recons_{str(information['id'])}", path_to_save=path_to_save)
        model.visual_reconstruction(v[0], add=f"original_{str(information['id'])}", path_to_save=path_to_save)
