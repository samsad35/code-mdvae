from mdvae import VQMDVAE, SpeechVQVAE, VisualVQVAE, MeadDataset
from mdvae import analysis_resynthesis, analysis_transformation_resynthesis
from hydra import initialize, compose
import unittest
from pathlib import Path

path = r"checkpoints/2022/mdvae-Y2022M3D17-12h0"


class TestEvaluation(unittest.TestCase):
    def load(self):
        initialize(config_path=f"{path}/config_mdvae")
        cfg = compose(config_name="config")

        """ Data """
        self.dataset = MeadDataset(root_modality_1=Path(r"D:\These\data\Audio-Visual\MEAD\Train"),
                                   root_modality_2=Path(r"D:\These\data\Audio-Visual\MEAD\Train_landmark_openface"),
                                   h5_speech_path=r"H5/speech_vq.hdf5",
                                   h5_visual_path="H5/visual_vq.hdf5",
                                   speaker_retain_test=[""],
                                   speaker_retain_validation=[""],
                                   train=False)

        print("=" * 100)
        """ VQ-VAE """
        self.speech_vqvae = SpeechVQVAE(**cfg.vqvae_1)
        self.speech_vqvae.load(path_model=r"checkpoints/VQVAE/speech/model_checkpoint_Y2022M3D5")

        self.visual_vqvae = VisualVQVAE(**cfg.vqvae_2)
        self.visual_vqvae.load(path_model=r"checkpoints/VQVAE/visual/model_checkpoint_Y2022M2D13")

        """ MDVAE """
        self.model = VQMDVAE(config_model=cfg.model, vqvae_speech=self.speech_vqvae, vqvae_visual=self.visual_vqvae)
        self.model.load_model(path_model=f"{path}/mdvae_model_checkpoint")
        self.model.to("cuda")
        print("=" * 100)

    def test_analysis_resynthesis(self):
        self.load()
        analysis_resynthesis(model=self.model,
                             dataset=self.dataset,
                             information=dict(id="W14", emotion="7", level="3", name="001"),
                             path_to_save=r"example/analysis_resynthesis")

    def test_analysis_transformation_resynthesis(self):
        self.load()
        analysis_transformation_resynthesis(model=self.model,
                                            dataset=self.dataset,
                                            target=dict(id="M03", emotion="7", level="3", name="015"),
                                            source=dict(id="W14", emotion="7", level="3", name="008"),
                                            variable_to_switch=["w"],
                                            path_to_save=r"example/analysis_transformation_resynthesis")


if __name__ == '__main__':
    unittest.main()
