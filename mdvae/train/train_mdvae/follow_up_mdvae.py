from datetime import datetime
import pandas
import torch
import pickle
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from .idr_torch import IDR


# from torch.utils.tensorboard import SummaryWriter


class Follow:
    def __init__(self, name: str, dir_save: str = "", multigpu_bool: bool = False):
        self.name = name
        self.datatime_start = datetime.today()
        self.dir_save = dir_save
        self.create_directory()
        self.table = {"epoch": [], "loss_train": [], "loss_validation": []}
        self.best_loss = 1e8
        self.multigpu_bool = multigpu_bool
        if self.multigpu_bool:
            self.idr = IDR()

    def create_directory(self):
        dir_sav = Path(self.dir_save) / self.name.upper()
        dir_sav.mkdir(exist_ok=True)
        to_day = str(self.datatime_start.date().year) + '-' + str(self.datatime_start.date().month) + '-' +\
                 str(self.datatime_start.date().day)
        time = str(self.datatime_start.time().hour) + '-' + str(self.datatime_start.time().minute)
        path_date = dir_sav / to_day
        path_date.mkdir(exist_ok=True)
        path_time = path_date / time
        path_time.mkdir(exist_ok=True)
        self.path = path_time
        shutil.copytree(r"config_mdvae", path_time / "config_mdvae", dirs_exist_ok=True)
        path_sample = path_time / "samples"
        self.path_samples = path_sample
        path_sample.mkdir(exist_ok=True)

    def find_best_model(self, loss_validation):
        if loss_validation <= self.best_loss:
            self.best_loss = loss_validation
            return True
        else:
            return False

    def save_model(self, boolean: bool, parameters: dict, epoch: int, every_step: int = 10):
        if epoch % every_step == 0:
            if self.multigpu_bool:
                if self.idr.local_rank == 0:
                    torch.save(parameters, f'{self.path}/model_checkpoint')
                    print(f"\t - Model saved: [loss:{parameters['loss']}]")
            else:
                torch.save(parameters, f'{self.path}/model_checkpoint')
                print(f"\t - Model saved: [loss:{parameters['loss']}]")
        if boolean:
            if self.multigpu_bool:
                if self.idr.local_rank == 0:
                    torch.save(parameters, f'{self.path}/model')
                    print(f"\t - Best Model saved: [loss:{parameters['loss']}]")
            else:
                torch.save(parameters, f'{self.path}/model')
                print(f"\t - Best Model saved: [loss:{parameters['loss']}]")

    def push(self, epoch: int, loss_train: float, loss_validation: float):
        self.table['epoch'].append(epoch)
        self.table['loss_train'].append(loss_train)
        self.table['loss_validation'].append(loss_validation)

    def save_csv(self):
        df = pandas.DataFrame(self.table)
        df.to_csv(path_or_buf=f'{self.path}/model_table.csv')

    def save_dict(self):
        a_file = open(f"{self.path}/table.pkl", "wb")
        pickle.dump(self.table, a_file)
        a_file.close()

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.table['epoch'], self.table['loss_train'], label="train", color="blue")
        plt.plot(self.table['epoch'], self.table['loss_validation'], label="validation", linestyle='dashed', color="red")
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Mean Loss', fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig(f'{self.path}/x_plot_loss.png')
        plt.savefig(f'{self.path}/x_plot_loss.svg')
        plt.close()

    def load_dict(self, path: str):
        a_file = open(f"{path}/table.pkl", "rb")
        self.table = pickle.load(a_file)

    def __call__(self, epoch: int, loss_train: float, loss_validation: float, parameters: dict, figure=None):
        self.push(epoch, loss_train, loss_validation)
        self.save_model(boolean=self.find_best_model(loss_validation), parameters=parameters, epoch=epoch, every_step=5)
        self.save_csv()
        self.save_dict()
        self.plot()
        if figure is not None:
            plt.savefig()
