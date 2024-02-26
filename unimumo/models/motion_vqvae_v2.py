import numpy as np
import torch
import pytorch_lightning as pl
from einops import rearrange
import typing as tp

from unimumo.util import instantiate_from_config
from unimumo.audio.audiocraft_.quantization.vq import ResidualVectorQuantizer
from unimumo.motion.motion_process import recover_from_ric
from unimumo.modules.motion_vqvae_module import Encoder, Decoder


class MotionVQVAE(pl.LightningModule):
    def __init__(
        self,
        motion_config: dict,
        quantizer_config: dict,
        loss_config: dict,
        music_key: str = "waveform",
        motion_key: str = "motion",
        monitor: tp.Optional[str] = None,
    ):
        super().__init__()
        self.motion_key = motion_key
        self.music_key = music_key

        self.motion_encoder = Encoder(**motion_config)
        self.motion_decoder = Decoder(**motion_config)
        self.quantizer = ResidualVectorQuantizer(**quantizer_config)

        self.loss = instantiate_from_config(loss_config)

        if monitor is not None:
            self.monitor = monitor

    def encode(self, x_motion: torch.Tensor) -> torch.Tensor:
        # x_music: [B, 1, 32000 x T], x_motion: [B, 20 x T, 263]
        assert x_motion.dim() == 3
        x_motion = rearrange(x_motion, 'b t d -> b d t')
        motion_emb = self.motion_encoder(x_motion)  # [B, 128, 50 x T]

        return motion_emb

    def decode(self, motion_emb: torch.Tensor) -> torch.Tensor:
        # music_emb: [B, 128, 50 x T], motion_emb: [B, 128, 50 x T]
        motion_recon = self.motion_decoder(motion_emb)
        motion_recon = rearrange(motion_recon, 'b d t -> b t d')  # [B, 20 x T, 263]

        return motion_recon

    def decode_from_code(self, motion_code: torch.Tensor):
        motion_emb = self.quantizer.decode(motion_code)
        return self.decode(motion_emb)

    def forward(self, batch: tp.Dict[str, torch.Tensor]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        motion_emb = self.encode(batch[self.motion_key])
        q_res_motion = self.quantizer(motion_emb, 50)
        motion_recon = self.decode(q_res_motion.x)

        return motion_recon, q_res_motion.penalty  # penalty is the commitment loss

    @staticmethod
    def motion_vec_to_joint(vec: torch.Tensor, motion_mean: np.ndarray, motion_std: np.ndarray) -> np.ndarray:
        # vec: [B, 20 x T, 263]
        mean = torch.tensor(motion_mean).to(vec)
        std = torch.tensor(motion_std).to(vec)
        vec = vec * std + mean
        joint = recover_from_ric(vec, joints_num=22)
        joint = joint.cpu().detach().numpy()
        return joint

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        motion_recon, commitment_loss = self.forward(batch)
        aeloss, log_dict_ae = self.loss(batch[self.motion_key], motion_recon, commitment_loss, split="train")

        for k, v in self.quantizer.state_dict().items():
            log_dict_ae[f'train/{k}'] = v.mean()

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        motion_recon, commitment_loss = self.forward(batch)
        aeloss, log_dict_ae = self.loss(batch[self.motion_key], motion_recon, commitment_loss, split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0)
        return [opt_ae], []

    @torch.no_grad()
    def log_videos(
        self, batch: tp.Dict[str, torch.Tensor], motion_mean: np.ndarray, motion_std: np.ndarray
    ) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        motion_recon, _ = self.forward(batch)
        waveform = batch[self.music_key].unsqueeze(1).detach().cpu().numpy()

        joint = self.motion_vec_to_joint(motion_recon, motion_mean, motion_std)
        gt_joint = self.motion_vec_to_joint(batch[self.motion_key], motion_mean, motion_std)
        return waveform, joint, gt_joint
