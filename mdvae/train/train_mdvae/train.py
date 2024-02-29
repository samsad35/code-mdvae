from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from ...base import Train
from ...model import VisualVQVAE, VQMDVAE, SpeechVQVAE
from ...tool import Monitor
import matplotlib.pyplot as plt
from .follow_up_mdvae import Follow
from .loss_function import Loss
from ...data import MeadDataset
from torchvision.utils import make_grid
import kornia as K
from scipy.io.wavfile import write
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .idr_torch import IDR
import librosa

torch.cuda.empty_cache()


class MdvaeTrainer(Train):
    def __init__(self, mdvae: VQMDVAE, vqvae_speech: SpeechVQVAE, vqvae_visual: VisualVQVAE,
                 training_data: MeadDataset, validation_data: MeadDataset,
                 config_training: dict = None,
                 audio_config: dict = None,
                 multigpu_bool: bool = False,
                 gpu_monitor: bool = False):
        super().__init__()
        if multigpu_bool:
            self.idr = IDR()
            dist.init_process_group(backend='nccl',
                                    init_method='env://',
                                    world_size=self.idr.size,
                                    rank=self.idr.rank)
            torch.cuda.set_device(self.idr.local_rank)

        self.device = torch.device(config_training['device'])
        """ Model """
        self.model = mdvae
        self.vqvae_speech = vqvae_speech
        self.vqvae_visual = vqvae_visual
        self.model = self.model.to(self.device)
        self.vqvae_speech.to(self.device)
        self.vqvae_visual.to(self.device)
        if multigpu_bool:
            self.model = DDP(self.model, device_ids=[self.idr.local_rank], find_unused_parameters=True)

        """ Dataloader """
        if multigpu_bool:
            train_sampler = torch.utils.data.distributed.DistributedSampler(training_data,
                                                                            num_replicas=self.idr.size,
                                                                            rank=self.idr.rank,
                                                                            shuffle=True)
            self.training_loader = torch.utils.data.DataLoader(dataset=training_data,
                                                               batch_size=config_training[
                                                                              'batch_size'] // self.idr.size,
                                                               shuffle=False,
                                                               num_workers=config_training['num_workers'],
                                                               pin_memory=True,
                                                               drop_last=True,
                                                               sampler=train_sampler)
            val_sampler = torch.utils.data.distributed.DistributedSampler(validation_data,
                                                                          num_replicas=self.idr.size,
                                                                          rank=self.idr.rank,
                                                                          shuffle=True)
            self.validation_loader = torch.utils.data.DataLoader(dataset=validation_data,
                                                                 batch_size=config_training[
                                                                                'batch_size'] // self.idr.size,
                                                                 shuffle=False,
                                                                 num_workers=0,
                                                                 pin_memory=True,
                                                                 drop_last=True,
                                                                 sampler=val_sampler,
                                                                 prefetch_factor=2)
        else:
            self.training_loader = DataLoader(training_data, batch_size=config_training['batch_size'], shuffle=True,
                                              num_workers=config_training['num_workers'], drop_last=True, )
            self.validation_loader = DataLoader(validation_data, batch_size=config_training['batch_size'], shuffle=True,
                                                num_workers=0, drop_last=True)

        """ Optimizer """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config_training['learning_rate'])
        # self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config_training['scheduler'][1],
                                                            gamma=config_training['scheduler'][2])

        """ Loss """
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.loss = Loss(n_wu=None)

        """ Config """
        self.config_training = config_training
        self.load_epoch = 0
        self.step_count = 0
        self.parameters = dict()
        self.h5_bool = training_data.h5_bool
        self.multigpu_bool = multigpu_bool
        self.beta = config_training['beta']
        self.win_length = int(audio_config['win_length'] * audio_config['sampling_rate'])
        self.hop = int(audio_config['hop_percent'] * self.win_length)
        self.n_fft = self.win_length

        """ Follow """
        self.follow = Follow("mdvae", dir_save=r"checkpoints", multigpu_bool=multigpu_bool)
        if gpu_monitor:
            self.gpu_monitor = Monitor(delay=60)
        else:
            self.gpu_monitor = None

    def one_epoch(self):
        self.model.train()
        losses = []
        for batch_idx, (modality_1, modality_2, _) in enumerate(tqdm(iter(self.training_loader))):
            self.optimizer.zero_grad()
            x_audio, x_visual = modality_1.to(self.device), modality_2.to(self.device)
            latent_space, x_audio_recons, x_visual_recons = self.model(x_audio, x_visual)
            loss, other = self.loss.loss_mdvae(x_audio, x_visual, x_audio_recons,
                                               x_visual_recons, latent_space, seq_length=x_audio.shape[1],
                                               batch_size=x_audio.shape[0])
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        self.plot_images(x_visual[0], save=f"{self.follow.path_samples}/train_original.png", show=False,
                         vqvae=self.vqvae_visual)
        self.plot_images(x_visual_recons[0], save=f"{self.follow.path_samples}/train_recons.png", show=False,
                         vqvae=self.vqvae_visual)
        return losses

    def fit(self):
        for e in range(self.load_epoch, self.config_training["total_epoch"]):
            if self.multigpu_bool:
                self.training_loader.sampler.set_epoch(e)
                self.validation_loader.sampler.set_epoch(e)
            losses = self.one_epoch()
            self.lr_scheduler.step()
            if e % 1 == 0:
                losses_val = self.eval()
                avg_loss_train = sum(losses) / len(losses)
                avg_loss_val = sum(losses_val) / len(losses_val)
                if self.multigpu_bool:
                    model_parameter = self.model.module.state_dict()
                else:
                    model_parameter = self.model.state_dict()
                self.parameters = dict(model=model_parameter,
                                       optimizer=self.optimizer.state_dict(),
                                       scheduler=self.lr_scheduler.state_dict(),
                                       epoch=e,
                                       loss=avg_loss_train)
                print(
                    f'In epoch {e}, average traning loss is {avg_loss_train}. '
                    f'and average validation loss is {avg_loss_val}')
                self.follow(epoch=e, loss_train=avg_loss_train, loss_validation=avg_loss_val,
                            parameters=self.parameters)
            if self.gpu_monitor is not None:
                self.gpu_monitor.stop()

    def griffin_lim(self, S, **kwargs):
        signal = librosa.griffinlim(S, hop_length=self.hop, win_length=self.win_length, **kwargs)
        return signal

    def save_wav(self, indices, save: str = None, vqvae: SpeechVQVAE = None):
        vq_output_eval = torch.reshape(indices, (-1, 8, 64))
        loss, quantized, perplexity, _ = vqvae._vq_vae(vq_output_eval)
        audio = vqvae._decoder(quantized)
        signal = self.griffin_lim(np.sqrt(torch.transpose(audio.squeeze(1), 0, 1).cpu().detach().numpy()))
        write(save, 16000, signal)

    @staticmethod
    def plot_images(indices, show: bool = True, save: str = None, vqvae: VisualVQVAE = None):
        vq_output_eval = torch.reshape(indices, (-1, 32, 8, 8))
        loss, quantized, perplexity, _ = vqvae._vq_vae(vq_output_eval)
        images = vqvae._decoder(quantized)
        plt.figure(figsize=(15, 15))
        out: torch.Tensor = make_grid(images + 0.5, nrow=10, padding=10)
        out_np: np.array = K.tensor_to_image(out)
        plt.imshow(out_np)
        plt.axis('off')
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()

    def eval(self):
        torch.cuda.empty_cache()
        self.model.eval()
        losses = []
        for batch_idx, (modality_1, modality_2, _) in enumerate(tqdm(iter(self.training_loader))):
            self.optimizer.zero_grad()
            x_audio, x_visual = modality_1.to(self.device), modality_2.to(self.device)
            latent_space, x_audio_recons, x_visual_recons = self.model(x_audio, x_visual)
            loss, other = self.loss.loss_mdvae(x_audio, x_visual, x_audio_recons,
                                               x_visual_recons, latent_space, seq_length=x_audio.shape[1],
                                               batch_size=x_audio.shape[0])
            losses.append(loss.item())
        return losses

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(path)
        if self.multigpu_bool:
            self.model.module.load_state_dict(checkpoint['model'])  # load checkpoint for multi-GPU
        else:
            checkpoint = torch.load(path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        self.load_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"model: ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]")

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
