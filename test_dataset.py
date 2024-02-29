from mdvae import MeadDataset
import torch
from pathlib import Path

torch.cuda.empty_cache()


def main():
    dataset = MeadDataset(root_modality_1=Path(r"D:\These\data\Audio-Visual\MEAD\Train"),
                          root_modality_2=Path(r"D:\These\data\Audio-Visual\MEAD\Train_landmark_openface"),
                          h5_speech_path=r"H5/speech_vq.hdf5",
                          h5_visual_path="H5/visual_vq.hdf5",
                          speaker_retain_test=["M03"],
                          speaker_retain_validation=[""],
                          train=False)

    print(dataset[0][0].shape)
    print(dataset[0][1].shape)


if __name__ == '__main__':
    main()
