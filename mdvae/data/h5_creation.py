from ..model import VisualVQVAE, SpeechVQVAE
from .dataset_mead import MeadDataset
import h5py
from ..tool import read_video_decord
from tqdm import tqdm
import torch
import librosa
import numpy as np
import torchvision.transforms as transforms

torch.cuda.empty_cache()
# win_length = int(64e-3 * 16000)
# n_fft = 1024
# hop = int(0.625 / 2 * win_length)
# win_length = win_length


def load_audio(file: str):
    wav, sr = librosa.load(path=file, sr=16000)
    wav = wav / np.max(np.abs(wav))
    return wav, sr


def stft(wave_audio, **kwargs) -> tuple:
    X = librosa.stft(wave_audio, **kwargs)  # pad_mode = 'reflected'
    magnitude = np.abs(X)
    phase = np.angle(X)
    return magnitude, phase


def get_input_audio(file, vqvae, device, **kwargs):
    signal, rate = load_audio(file)
    magnitude, phase = stft(signal, **kwargs)
    spectro = magnitude ** 2
    spectro = torch.from_numpy(spectro.transpose()).to(device)[:, None, :]
    vq_output_eval = vqvae._pre_vq_conv(vqvae._encoder(spectro))
    vq_output_eval = vq_output_eval.reshape(-1, 64 * 8)
    return vq_output_eval.cpu().detach().numpy()


def get_input_visual(file,
                     vqvae,
                     device,
                     transform):
    images = read_video_decord(file_path=file)
    temps = torch.tensor([])
    for image in images:
        image = transform(image)
        temps = torch.cat((temps, image[None]), dim=0)
    vq_output_eval = vqvae._pre_vq_conv(vqvae._encoder(temps.to(device)))
    # vq_output_eval = vq_output_eval.reshape(-1, 512)
    vq_output_eval = vq_output_eval.reshape(-1, 32 * 8 * 8)
    return vq_output_eval.cpu().detach().numpy()


def h5_creation(vqvae,
                table,
                dir_save: str,
                audio_config: dict,
                visual_config: dict,
                audio_bool: bool = False,
                device: str = "cuda"):
    file_h5 = h5py.File(dir_save, 'a')
    vqvae.to('cuda')
    if audio_bool:
        suff = "audio_"
    else:
        suff = "visual_"
    device = torch.device(device)
    # ------------------------
    if audio_bool:
        win_length = int(audio_config['win_length'] * audio_config['sampling_rate'])
        hop = int(audio_config['hop_percent'] * win_length)
        n_fft = win_length
        stft_config = dict(n_fft=n_fft, hop_length=hop, win_length=win_length)
    else:
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                                        transforms.Resize(64)])
    # ------------------------
    with tqdm(total=len(table)) as pbar:
        for i in range(len(table)):
            id = table.iloc[i]["id"]
            name = table.iloc[i]["name"]
            level = table.iloc[i]["level"]
            emotion = table.iloc[i]["emotion"]
            path_audio = table.iloc[i]["path_audio"]
            path_visual = table.iloc[i]["path_visual"]
            pbar.update(1)
            pbar.set_description(f"ID: {id}, name: {name}, emotion: {emotion}")
            # Get indices for each file .mp4
            if file_h5.get(f'/{id}/{emotion}/{level}/{suff}{name}'):
                continue
            if audio_bool:
                data = get_input_audio(file=path_audio, vqvae=vqvae, device=device, **stft_config)
            else:
                data = get_input_visual(file=path_visual, vqvae=vqvae, device=device, transform=transform)
            # Create the path in H5 file:
            if not file_h5.get(f'/{id}'):
                file_h5.create_group(f'/{id}')
            if not file_h5.get(f'/{id}/{emotion}'):
                file_h5.create_group(f'/{id}/{emotion}')
            if not file_h5.get(f'/{id}/{emotion}/{level}'):
                file_h5.create_group(f'/{id}/{emotion}/{level}')
            group_temp = file_h5[f'/{id}/{emotion}/{level}']
            # Save indices in H5 file
            image_h5 = group_temp.create_dataset(name=f"{suff}{name}", data=data, dtype="float32")
            image_h5.attrs.create('id', id)

    # Close h5 file
    file_h5.flush()
    file_h5.close()
