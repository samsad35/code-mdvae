from mdvae import VQMDVAE, SpeechVQVAE, VisualVQVAE, MdvaeTrainer, MeadDataset
import hydra
import os
from omegaconf import DictConfig
import torch
from pathlib import Path
torch.cuda.empty_cache()


@hydra.main(config_path=f"config_mdvae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    dataset_train = MeadDataset(root_modality_1=Path(r"..."),
                                root_modality_2=Path(r"..."),
                                h5_speech_path=r"H5/speech_vq.hdf5",
                                h5_visual_path="H5/visual_vq.hdf5",
                                speaker_retain_test=["..."],
                                speaker_retain_validation=[""],
                                train=True)

    dataset_validation = MeadDataset(root_modality_1=Path(r"..."),
                                     root_modality_2=Path(r"..."),
                                     h5_speech_path=r"H5/speech_vq.hdf5",
                                     h5_visual_path="H5/visual_vq.hdf5",
                                     speaker_retain_test=["..."],
                                     speaker_retain_validation=[""],
                                     train=False)

    print("=" * 100)
    """ VQ-VAE """
    speech_vqvae = SpeechVQVAE(**cfg.vqvae_1)
    speech_vqvae.load(path_model=r"checkpoints/VQVAE/speech/model_checkpoint_Y2022M3D5")

    visual_vqvae = VisualVQVAE(**cfg.vqvae_2)
    visual_vqvae.load(path_model=r"checkpoints/VQVAE/visual/model_checkpoint_Y2022M2D13")

    """ MDVAE """
    model = VQMDVAE(config_model=cfg.model, vqvae_speech=speech_vqvae, vqvae_visual=visual_vqvae)
    print("=" * 100)

    """ Trainer """
    trainer = MdvaeTrainer(mdvae=model, vqvae_speech=speech_vqvae, vqvae_visual=visual_vqvae,
                           training_data=dataset_train, validation_data=dataset_validation,
                           config_training=cfg.training_config, audio_config=cfg.audio_config)
    # trainer.load(path=r"...")
    trainer.fit()


if __name__ == '__main__':
    main()
