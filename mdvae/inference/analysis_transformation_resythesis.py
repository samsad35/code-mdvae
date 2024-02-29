import torch
from ..model import VQMDVAE
from ..data import MeadDataset


def analysis_transformation_resynthesis(model: VQMDVAE,
                                        dataset: MeadDataset,
                                        source: dict = None,
                                        target: dict = None,
                                        variable_to_switch: list = None,
                                        seq_length: int = 60,
                                        path_to_save: str = None):
    a_source, v_source = dataset.read(**source)
    a_target, v_target = dataset.read(**target)

    assert a_source.shape[0] > seq_length, print("Problem with sequence length !")
    assert a_target.shape[0] > seq_length, print("Problem with sequence length !")

    a_source, v_source = torch.from_numpy(a_source[:seq_length]).unsqueeze(0).to("cuda"), \
                         torch.from_numpy(v_source[:seq_length]).unsqueeze(0).to("cuda")
    a_target, v_target = torch.from_numpy(a_target[:seq_length]).unsqueeze(0).to("cuda"), \
                         torch.from_numpy(v_target[:seq_length]).unsqueeze(0).to("cuda")

    latent_space_source = model.encoder(a_source, v_source)
    model.visual_reconstruction(v_source[0], add="0_source", path_to_save=path_to_save)

    latent_space_target = model.encoder(a_target, v_target)
    model.visual_reconstruction(v_target[0], add="1_target", path_to_save=path_to_save)

    new_latent_space = dict(zaudio=latent_space_source["zaudio"][0], zvisual=latent_space_source["zvisual"][0],
                            w=latent_space_source["w"][0], zav=latent_space_source["zav"][0])
    for variable in variable_to_switch:
        new_latent_space[variable] = latent_space_target[variable][0]
    a_tran, v_tran = model.decoder(**new_latent_space)
    model.visual_reconstruction(v_tran[0], add="0_transformed", path_to_save=path_to_save)
