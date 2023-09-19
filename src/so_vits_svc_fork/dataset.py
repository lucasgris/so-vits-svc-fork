from __future__ import annotations

import random
from pathlib import Path
from random import Random
from typing import Sequence

import os
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torch
import torchaudio

from .hparams import HParams
from .preprocessing.preprocess_utils import check_hubert_min_duration
from .modules.mel_processing import spec_to_mel_torch, spectrogram_torch

from so_vits_svc_fork.f0 import compute_f0, interpolate_f0

from logging import getLogger
LOG = getLogger(__name__)

class TextAudioDataset(Dataset):
    def __init__(self, hps: HParams, files):
        self.datapaths = [
            Path(x).parent / (Path(x).name)
            for x in Path(files)
            .read_text("utf-8")
            .splitlines()
        ]
        self.hps = hps
        self.random = Random(hps.train.seed)
        self.random.shuffle(self.datapaths)
        self.max_spec_len = 800

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        try:
            audio, sr = torchaudio.load(self.datapaths[index])
            if sr != self.hps.data.sampling_rate:
                audio = torchaudio.functional.resample(audio, sr, self.hps.data.sampling_rate)

            if audio.shape[1]/sr < 0.3:
                LOG.warning(f"{self.datapaths[index]} is too short. Padding it.")
                padding = torch.zeros((audio.shape[0], int(sr * 0.4)-audio.shape[1]))
                audio = torch.cat([audio, padding], dim=-1)
            
            # Compute f0
            if os.path.isfile(str(self.datapaths[index])+".pitch.pt") and self.hps.data.f0_method == "crepe":
                pitch = torch.load(str(self.datapaths[index])+".pitch.pt")
                f0, uv = pitch["f0"], pitch["uv"]
            else:
                f0 = compute_f0(
                    audio.squeeze(0).cpu().numpy(), 
                    sampling_rate=self.hps.data.sampling_rate, 
                    hop_length=self.hps.data.hop_length, 
                    method=self.hps.data.f0_method, 
                    device='cpu'
                )
                f0, uv = interpolate_f0(f0)
                f0 = torch.from_numpy(f0).float()
                uv = torch.from_numpy(uv).float()
                torch.save({"f0": f0, "uv": uv}, str(self.datapaths[index])+".pitch.pt")
            
            spec = spectrogram_torch(audio, self.hps).squeeze(0)
            mel_spec = spec_to_mel_torch(spec, self.hps)

            # cut long data randomly
            spec_len = mel_spec.shape[1]
            hop_len = self.hps.data.hop_length
            if spec_len > self.max_spec_len:
                start = self.random.randint(0, spec_len - self.max_spec_len)
                end = start + self.max_spec_len - 10
                
                audio = audio[:, start * hop_len : end * hop_len]
                spec = spec[..., start:end]
                mel_spec = mel_spec[..., start:end]
                f0 = f0[..., start:end]
                uv = uv[..., start:end]

            spk_name = self.datapaths[index].parent.name
            spk_id = self.hps.spk.__dict__[spk_name]
            spk_id = torch.tensor(spk_id).long()

            data = {
                "spec": spec,
                "mel_spec": mel_spec,
                "f0": f0,
                "uv": uv,
                "audio": audio,
                "spk": spk_id,
            }
            
            data = {k: v.cpu() for k, v in data.items()}

            return data
        except Exception as e:
            LOG.error(f"Error loading data {self.datapaths[index]} {str(e)}")
            with open('errors.txt', 'a') as f:
                print(self.datapaths[index], file=f)
            self.datapaths.pop(index)
            return self.__getitem__(random.randint(0, len(self)))

    def __len__(self) -> int:
        return len(self.datapaths)


def _pad_stack(array: Sequence[torch.Tensor]) -> torch.Tensor:
    max_idx = torch.argmax(torch.tensor([x_.shape[-1] for x_ in array]))
    max_x = array[max_idx]
    x_padded = [
        F.pad(x_, (0, max_x.shape[-1] - x_.shape[-1]), mode="constant", value=0)
        for x_ in array
    ]
    return torch.stack(x_padded)


class TextAudioCollate(nn.Module):
    def forward(
        self, batch: Sequence[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, ...]:
        batch = [b for b in batch if b is not None]
        batch = list(sorted(batch, key=lambda x: x["mel_spec"].shape[1], reverse=True))
        lengths = torch.tensor([b["mel_spec"].shape[1] for b in batch]).long()
        results = {}
        for key in batch[0].keys():
            if key not in ["spk"]:
                results[key] = _pad_stack([b[key] for b in batch]).cpu()
            else:
                results[key] = torch.tensor([[b[key]] for b in batch]).cpu()

        return (
            results["f0"],
            results["spec"],
            results["mel_spec"],
            results["audio"],
            results["spk"],
            lengths,
            results["uv"],
        )
