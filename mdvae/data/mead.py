import os
from tqdm import tqdm
import glob
import pandas
import numpy as np
from pathlib import Path


class MEAD:
    def __init__(self,
                 root_modality_1: Path = None,
                 root_modality_2: Path = None,
                 ext="mp4",
                 view: str = "front",
                 audiovisual_bool: bool = True
                 ):
        self.root_modality_1 = root_modality_1
        self.root_modality_2 = root_modality_2
        self.length = len(glob.glob(f"{root_modality_2}/**/video/{view}/**/**/*.{ext}"))
        self.emotion_ = dict(angry=0, contempt=1, disgusted=2, fear=3, happy=4, neutral=5, sad=6, surprised=7)
        self.level_ = dict(level_1=1, level_2=2, level_3=3)
        self.table = None
        self.ext = ext
        self.audiovisual_bool = audiovisual_bool

    @staticmethod
    def __generator__(directory: Path):
        all_dir = os.listdir(directory)
        for d in all_dir:
            yield d, directory / d

    def generator(self):
        for id, id_root in self.__generator__(self.root_modality_2):
            for emotion, emotion_root in self.__generator__(id_root / "video/front"):
                for level, level_root in self.__generator__(emotion_root):
                    for name, path_visual in self.__generator__(level_root):
                        if self.ext in name:
                            path_audio = str(path_visual).replace(str(self.root_modality_2), str(self.root_modality_1))
                            if self.audiovisual_bool:
                                path_audio = str(path_audio).replace(r"video\front", "audio")
                                path_audio = str(path_audio).replace(".mp4", ".m4a")
                            yield id, name.split(".")[0], Path(path_audio), path_visual, self.emotion_[emotion], self.level_[level]

    def generate_table(self):
        files_list_audio = []
        files_list_video = []
        id_list = []
        name_list = []
        emotion_list = []
        level_list = []
        with tqdm(total=self.length, desc=f"Create table (MEAD): ") as pbar:
            for id, name, path_audio, path_visual, emotion, level in self.generator():
                files_list_audio.append(path_audio)
                files_list_video.append(path_visual)
                id_list.append(id)
                emotion_list.append(emotion)
                name_list.append(name)
                level_list.append(level)
                pbar.update(1)
        self.table = pandas.DataFrame(
            np.array([id_list, name_list, files_list_audio, files_list_video, emotion_list, level_list]).transpose(),
            columns=['id', 'name', 'path_audio', 'path_visual', 'emotion', 'level'])


if __name__ == '__main__':
    mead = MEAD(root_modality_1=Path(r"D:\These\data\Audio-Visual\MEAD\Train"),
                root_modality_2=Path(r"D:\These\data\Audio-Visual\MEAD\Train_landmark_openface")
                )
    mead.generate_table()
    print(mead.table)
