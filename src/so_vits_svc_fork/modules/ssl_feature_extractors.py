import torch
from torch import nn

import torchaudio
from transformers import HubertModel

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

class HubertFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path, freeze=True, svc_model_sr=16000, legacy_final_proj=True):
        super().__init__()
        self.extractor_sr = 16000
        self.svc_model_sr = svc_model_sr
        self.freeze = freeze
        self.legacy_final_proj = legacy_final_proj
        if self.legacy_final_proj:
            self.model = HubertModelWithFinalProj.from_pretrained(checkpoint_path)
        else:
            self.model = HubertModel.from_pretrained(checkpoint_path)
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        else:
            self.train()

    def extract_features(self, y, **kwargs):
        y = y.squeeze()
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if self.svc_model_sr != self.extractor_sr:
            y = torchaudio.functional.resample(
                y,
                orig_freq=self.svc_model_sr,
                new_freq=self.extractor_sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )

        if self.freeze:
            with torch.no_grad():
                if self.legacy_final_proj:
                    c = self.model(y, output_hidden_states=True)["hidden_states"][9]
                    c = self.model.final_proj(c)
                else:
                    c = self.model(y)["last_hidden_state"]
        else:
            if self.legacy_final_proj:
                c = self.model(y, output_hidden_states=True)["hidden_states"][9]
                c = self.model.final_proj(c)
            else:    
                c = self.model(y)["last_hidden_state"]
        return c.transpose(1, 2)
