from mdvae import VQMDVAE, SpeechVQVAE, VisualVQVAE
import hydra
import os
from omegaconf import DictConfig
import torch

torch.cuda.empty_cache()

path = r"checkpoints/2022/mdvae-Y2022M3D17-12h0"


@hydra.main(config_path=f"{path}/config_mdvae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    print("=" * 100)
    """ VQ-VAE """
    speech_vqvae = SpeechVQVAE(**cfg.vqvae_1)
    speech_vqvae.load(path_model=r"checkpoints/VQVAE/speech/model_checkpoint_Y2022M3D5")

    visual_vqvae = VisualVQVAE(**cfg.vqvae_2)
    visual_vqvae.load(path_model=r"checkpoints/VQVAE/visual/model_checkpoint_Y2022M2D13")

    """ MDVAE """
    model = VQMDVAE(config_model=cfg.model, vqvae_speech=speech_vqvae, vqvae_visual=visual_vqvae)
    model.load_model(path_model=f"{path}/mdvae_model_checkpoint")
    print("=" * 100)


if __name__ == '__main__':
    main()
