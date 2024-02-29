from mdvae import h5_creation, MeadDataset, SpeechVQVAE, VisualVQVAE
import hydra
import os
from omegaconf import DictConfig
import torch
from pathlib import Path

torch.cuda.empty_cache()


@hydra.main(config_path=f"config_mdvae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """
    dataset = MeadDataset(root_modality_1=Path(r"..."),
                          root_modality_2=Path(r"..."),
                          h5_speech_path="",
                          h5_visual_path="",
                          speaker_retain_test=["..."],
                          speaker_retain_validation=["..."],
                          train=False)

    """ VQ-VAE """
    speech_vqvae = SpeechVQVAE(**cfg.vqvae_1)
    speech_vqvae.load(path_model=r"checkpoints/VQVAE/speech/model_checkpoint_Y2022M3D5")

    visual_vqvae = VisualVQVAE(**cfg.vqvae_2)
    visual_vqvae.load(path_model=r"checkpoints/VQVAE/visual/model_checkpoint_Y2022M2D13")

    """ H5 creation """
    h5_creation(vqvae=visual_vqvae,
                table=dataset.table,
                dir_save=r"H5/visual_vq.hdf5",
                audio_config=cfg.audio_config,
                visual_config=cfg.visual_config,
                audio_bool=False)

    h5_creation(vqvae=speech_vqvae,
                table=dataset.table,
                dir_save=r"H5/audio_vq.hdf5",
                audio_config=cfg.audio_config,
                visual_config=cfg.visual_config,
                audio_bool=True)


if __name__ == '__main__':
    main()
