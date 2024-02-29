from .mead import MEAD
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
import torch


class MeadDataset(Dataset):
    def __init__(self,
                 root_modality_1: Path,
                 root_modality_2: Path,
                 h5_speech_path: str,
                 h5_visual_path: str,
                 speaker_retain_test: list = None,
                 speaker_retain_validation: list = None,
                 seq_length: int = 30,
                 train: bool = True):
        self.mead = MEAD(root_modality_1=root_modality_1, root_modality_2=root_modality_2)
        self.mead.generate_table()
        self.table = self.mead.table
        self.read_from_h5_bool = True
        # --------------------------------------------------------
        if (speaker_retain_test is not None) or (speaker_retain_validation is not None):
            self.speaker_retain = speaker_retain_test + speaker_retain_validation
            self.train = train
            self.table_()
        # --------------------------------------------------------
        self.index_wav = 0
        self.number_frames = 0
        self.current_frame = 0
        self.seq_length = seq_length
        self.train = train
        self.h5_bool = True if h5_visual_path is not None else False
        self.h5_speech_path = h5_speech_path
        self.h5_visual_path = h5_visual_path
        self.__len__()
        self.open()

    def table_(self):
        if not self.train:
            self.table = self.table.loc[self.table['id'].isin(self.speaker_retain)].reset_index(drop=True)
        else:
            self.table = self.table.loc[~self.table['id'].isin(self.speaker_retain)].reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def save_table(self, path: str):
        self.table.to_pickle(path)

    def get_information(self, index):
        id = self.table.iloc[index]['id']
        level = self.table.iloc[index]['level']
        emotion = self.table.iloc[index]['emotion']
        name = self.table.iloc[index]['name']
        return id, emotion, level, name

    @staticmethod
    def padding(data, seq_length=50):
        """

        :param seq_length:
        :param data:
        :return:
        """
        if len(data.shape) == 2:
            data = np.pad(data, ((0, seq_length - data.shape[0]), (0, 0)), 'wrap')
        return data

    def open(self):
        self.h5_speech = h5py.File(self.h5_speech_path, mode='r')
        self.h5_visual = h5py.File(self.h5_visual_path, mode='r')

    def read(self, id, emotion, level, name):
        a = np.array(self.h5_speech[f'/{id}/{emotion}/{level}/audio_{name}'])
        v = np.array(self.h5_visual[f'/{id}/{emotion}/{level}/visual_{name}'])
        return a, v

    def __getitem__(self, item):
        if not hasattr(self, 'h5_speech'):
            self.open()
        id, emotion, level, name = self.get_information(item)
        a, v = self.read(id, emotion, level, name)
        number_frames = v.shape[0]
        if number_frames < self.seq_length:
            a = self.padding(a, seq_length=self.seq_length)
            v = self.padding(v, seq_length=self.seq_length)
        number_frames = v.shape[0]
        a = torch.from_numpy(a)
        v = torch.from_numpy(v)
        current_frame = np.random.randint(0, number_frames - self.seq_length - 1)
        self.i_1 = a[current_frame:current_frame + self.seq_length]
        self.i_2 = v[current_frame:current_frame + self.seq_length]
        return self.i_1.type(torch.FloatTensor), self.i_2.type(torch.FloatTensor), emotion

