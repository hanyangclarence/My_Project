import os
import typing as tp
import warnings
import flashy.distrib
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import random
from collections import OrderedDict
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from unimumo.util import instantiate_from_config
from unimumo.audio.audiocraft_.models.mm_lm import LMModel, ConditionTensors
from unimumo.audio.audiocraft_.models.loaders import load_mm_lm_model
from unimumo.audio.audiocraft_.modules.conditioners import ConditioningAttributes, WavCondition
from unimumo.models.text_generation_model import TextGenerator


# backward compatible names mapping
_HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}

# trainable keys in music-to-motion pretraining
# this includes motion codebook, motion feed-forward and motion classification head
trainable_keys = [
    'motion_emb',
    'motion_linears',
    'linear1_motion',
    'linear2_motion',
    'norm1_motion',
    'norm2_motion'
]


class MusicMotionTransformer(pl.LightningModule):
    def __init__(
        self,
        name: str,
        music_key: str,
        motion_key: str,
        text_cond_key: str,
        motion_weight: float,
        length_single_modal: int,
        text_model_config: dict,
        feature_frame_rate: int = 50,

        stage: tp.Optional[str] = None,
        mm_ckpt: tp.Optional[str] = None,
        is_pretraining: bool = False,

        generation_params: tp.Optional[dict] = None,
        scheduler_config: tp.Optional[dict] = None,
        optimization_config: tp.Optional[dict] = None,

        monitor=None,
        debug: bool = False
    ):
        super().__init__()

        self.music_key = music_key
        self.motion_key = motion_key
        self.text_cond_key = text_cond_key

        self.motion_weight = motion_weight

        # load music motion transformer
        self.model: LMModel = self.get_pretrained_lm(name, use_autocast=False, debug=debug)

        # load music motion captioner
        self.text_model: TextGenerator = instantiate_from_config(text_model_config)

        # setup training stage and trainable parameters
        self.is_pretraining = is_pretraining
        assert stage in ['train_music_motion', 'train_caption'] or stage is None
        self.stage = stage
        if self.stage == 'train_music_motion':
            if self.is_pretraining:
                print('Pretrain on motion generation!')
            else:
                print('Finetune on music-motion joint generation!')
                self.init_music_motion_lm_with_pretrained(mm_ckpt)
        if self.stage == 'train_caption':
            print('In training caption stage!')
            self.init_music_motion_lm_with_pretrained(mm_ckpt)
        # freeze corresponding parameters
        self.setup_trainable_parameters()

        self.duration = generation_params.pop('duration')
        self.feature_frame_rate = feature_frame_rate
        self.sample_rate = 32000
        self.generation_params = generation_params

        self.max_sequence_length = (length_single_modal + self.model.n_q) * 2

        self.scheduler_config = scheduler_config
        self.optimization_config = optimization_config

        # set to manual backward in training step
        self.automatic_optimization = False

        if monitor is not None:
            self.monitor = monitor

    def get_pretrained_lm(
        self,
        name: str = 'facebook/musicgen-melody',
        device=None, use_autocast=False, debug=False
    ) -> LMModel:
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'
        print(f'Load lm and conditioner to {device}')

        if name in _HF_MODEL_CHECKPOINTS_MAP:
            warnings.warn(
                "MusicGen pretrained model relying on deprecated checkpoint mapping. " +
                f"Please use full pre-trained id instead: facebook/musicgen-{name}")
            name = _HF_MODEL_CHECKPOINTS_MAP[name]

        lm = load_mm_lm_model(name, device=device, use_autocast=use_autocast, debug=debug)
        if 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return lm

    def init_music_motion_lm_with_pretrained(self, ckpt: str):
        assert os.path.exists(ckpt), f'The provided path {ckpt} does not exist!'
        # load the music motion lm part of the ckpt
        pretrained_sd = torch.load(ckpt, map_location='cpu')['state_dict']
        mm_lm_sd = {k: v for k, v in pretrained_sd.items() if k.startswith("model.")}  # find keys with prefix "model."
        mm_lm_sd = {k[len("model."):]: v for k, v in mm_lm_sd.items()}  # remove the prefix "model."
        self.model.load_state_dict(mm_lm_sd)

    def setup_trainable_parameters(self):
        if self.stage == 'train_music_motion':
            if self.is_pretraining:
                # allow motion related parameters trainable
                for name, parameter in self.model.named_parameters():
                    if any([s in name for s in trainable_keys]):
                        parameter.requires_grad = True
                    else:
                        parameter.requires_grad = False
                # freeze all parameters for text generation model
                for name, parameter in self.text_model.named_parameters():
                    parameter.requires_grad = False
            else:
                # set all parameters in music motion transformer as trainable, except for condition provider
                for name, parameter in self.model.named_parameters():
                    parameter.requires_grad = True
                for name, parameter in self.model.condition_provider.named_parameters():
                    parameter.requires_grad = False
                # freeze all parameters for text generation model
                for name, parameter in self.text_model.named_parameters():
                    parameter.requires_grad = False
        elif self.stage == 'train_caption':
            # freeze all parameters in music-motion transformer model
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad = False
            # train all parameters in text generation model
            for name, parameter in self.text_model.named_parameters():
                parameter.requires_grad = True
        else:
            ValueError('Wrong stage settings!!')

    @rank_zero_only
    def print_trainable_parameters(self):
        trainable_name_list = []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                trainable_name_list.append(name)
        # remove repetitive names
        filtered_name = []
        for name in trainable_name_list:
            name = name.split('.')
            name = [s for s in name if not s.isdigit()]
            name = '.'.join(name)
            filtered_name.append(name)
        name_set = list(OrderedDict.fromkeys(filtered_name))
        name_count = {}
        for name in name_set:
            name_count[name] = sum([s == name for s in filtered_name])
        print('All trainable parameters:')
        for name, count in name_count.items():
            print(f'[{name}] x {count}')

    def training_step(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        batch_idx: int
    ):
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

        if self.stage == 'train_music_motion':  # train the music motion lm
            # # randomly choose the mode on this training step
            # mode = random.choice(['music_motion', 'music2motion', 'motion2music'])
            if self.is_pretraining:
                mode = 'music2motion'
            else:
                mode = random.choice(['music_motion', 'music2motion', 'motion2music'])
            text_condition = self.prepare_text_condition(text_cond, mode)

            music_output, motion_output = self.model.compute_predictions(
                music_code, motion_code, mode, [], condition_tensors=text_condition
            )
            music_logits, music_mask = music_output.logits, music_output.mask
            motion_logits, motion_mask = motion_output.logits, motion_output.mask

            music_loss, music_loss_per_codebook = self.compute_cross_entropy(music_logits, music_code, music_mask)
            motion_loss, motion_loss_per_codebook = self.compute_cross_entropy(motion_logits, motion_code, motion_mask)
            total_loss = music_loss * (1 - self.motion_weight) + motion_loss * self.motion_weight

            self.log("train/loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log(f"train/{mode}_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log(f"train/{mode}_music_loss", music_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log(f"train/{mode}_motion_loss", motion_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

            self.log(f'train/attn_weight', self.model.attention_weight.item(), prog_bar=True, logger=True, on_step=True, on_epoch=False)

            log_dict = {}
            # for k in range(len(music_loss_per_codebook)):
            #     log_dict[f'train/music_ce_q{k + 1}'] = music_loss_per_codebook[k]
            #     log_dict[f'train/motion_ce_q{k + 1}'] = motion_loss_per_codebook[k]

            optimizer = self.optimizers().optimizer
            lr_scheduler = self.lr_schedulers()

            if self.optimization_config['eager_sync']:
                with flashy.distrib.eager_sync_model(self.model):
                    self.manual_backward(total_loss)
            else:
                self.manual_backward(total_loss)
                flashy.distrib.sync_model(self.model)

            if self.optimization_config['max_norm']:
                log_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.optimization_config['max_norm']
                )

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

            for i in range(self.model.n_q):
                log_dict[f'train/music_emb_{i}'] = self.model.emb.state_dict()[f'{i}.weight'].mean().item()
                log_dict[f'train/motion_emb_{i}'] = self.model.motion_emb.state_dict()[f'{i}.weight'].mean().item()

            self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        else:  # train the text generation model
            batch_size = len(text_cond)

            # use null condition for music motion network
            descriptions: tp.List[str] = [
                '<music_prompt_start> <music_prompt_end> <motion_prompt_start> <motion_prompt_end>'
            ] * batch_size
            null_text_condition = self.prepare_text_condition(descriptions, mode='music_motion')

            # get music motion features using music motion LM
            with torch.no_grad():
                self.model.eval()
                music_motion_context = self.model.get_music_motion_context(
                    music_code, motion_code, [], condition_tensors=null_text_condition
                )

            text_loss = self.text_model(text_cond, music_motion_context)

            self.log("train/text_loss", text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

            optimizer = self.optimizers().optimizer
            lr_scheduler = self.lr_schedulers()

            self.manual_backward(text_loss)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

    def validation_step(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        batch_idx: int
    ):
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

        if self.stage == 'train_music_motion':
            # mode = random.choice(['music_motion', 'music2motion', 'motion2music'])
            if self.is_pretraining:
                mode = 'music2motion'
            else:
                mode = 'music_motion'
            text_condition = self.prepare_text_condition(text_cond, mode)

            music_output, motion_output = self.model.compute_predictions(
                music_code, motion_code, mode, [], condition_tensors=text_condition
            )
            music_logits, music_mask = music_output.logits, music_output.mask
            motion_logits, motion_mask = motion_output.logits, motion_output.mask

            music_loss, music_loss_per_codebook = self.compute_cross_entropy(music_logits, music_code, music_mask)
            motion_loss, motion_loss_per_codebook = self.compute_cross_entropy(motion_logits, motion_code, motion_mask)
            total_loss = music_loss * (1 - self.motion_weight) + motion_loss * self.motion_weight

            self.log("val/loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log(f"val/{mode}_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log(f"val/{mode}_music_loss", music_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log(f"val/{mode}_motion_loss", motion_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            # log_dict = {}
            # for k in range(len(music_loss_per_codebook)):
            #     log_dict[f'val/music_ce_q{k + 1}'] = music_loss_per_codebook[k]
            #     log_dict[f'val/motion_ce_q{k + 1}'] = motion_loss_per_codebook[k]
            #
            # self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        else:
            batch_size = len(text_cond)

            # use null condition for music motion network
            descriptions: tp.List[str] = [
                '<music_prompt_start> <music_prompt_end> <motion_prompt_start> <motion_prompt_end>'
            ] * batch_size
            null_text_condition = self.prepare_text_condition(descriptions, mode='music_motion')

            # get music motion features using music motion LM
            with torch.no_grad():
                self.model.eval()
                music_motion_context = self.model.get_music_motion_context(
                    music_code, motion_code, [], condition_tensors=null_text_condition
                )

            text_loss = self.text_model(text_cond, music_motion_context)

            self.log("val/text_loss", text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.LongTensor, mask: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def prepare_text_condition(self, descriptions: tp.List[str], mode: str) -> ConditionTensors:
        if mode == 'music2motion':
            dropped_out_descriptions = []
            # remove music descriptions
            for desc in descriptions:
                motion_description = desc.split('<music_prompt_end> ')[-1]
                dropped_out_descriptions.append(
                    '<music_prompt_start> <music_prompt_end> ' + motion_description
                )
            descriptions = dropped_out_descriptions
        elif mode == 'motion2music':
            dropped_out_descriptions = []
            # remove motion descriptions
            for desc in descriptions:
                music_description = desc.split(' <motion_prompt_start>')[0]
                dropped_out_descriptions.append(
                    music_description + ' <motion_prompt_start> <motion_prompt_end>'
                )
            descriptions = dropped_out_descriptions
        else:
            assert mode == 'music_motion'

        attributes = [ConditioningAttributes(text={'description': description}) for description in descriptions]

        attributes = self.model.cfg_dropout(attributes)

        attributes = self.model.att_dropout(attributes)

        # print drop out results for debug
        print(f"{mode}: {self.model.training}, {attributes[0].text['description']}")

        tokenized = self.model.condition_provider.tokenize(attributes, device=self.device)
        condition_tensors = self.model.condition_provider(tokenized)

        return condition_tensors

    def generate_sample(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        duration: tp.Optional[float] = None,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: tp.Optional[float] = None,
        return_result_only: bool = False
    ):
        attributes = self._prepare_tokens_and_attributes(batch[self.text_cond_key], mode='music_motion')

        music_gen, motion_gen = self._generate_tokens(
            attributes, mode='music_motion', duration=duration, temperature=temperature,
            conditional_guidance_scale=conditional_guidance_scale
        )
        if return_result_only:
            return music_gen, motion_gen
        else:
            return music_gen, motion_gen, batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

    def generate_single_modality(
        self,
        music_code: tp.Optional[torch.LongTensor] = None,  # (B, K, S)
        motion_code: tp.Optional[torch.LongTensor] = None,  # (B, K, S)
        text_description: tp.Optional[tp.List[str]] = None,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: tp.Optional[float] = None,
    ) -> torch.LongTensor:
        assert (music_code is None) ^ (motion_code is None), "Only one modality should be given."
        batch_size = music_code.shape[0] if music_code is not None else motion_code.shape[0]
        sequence_length = music_code.shape[-1] if music_code is not None else motion_code.shape[-1]
        mode = 'music2motion' if music_code is not None else 'motion2music'
        if text_description is None:
            text_description = [
                '<music_prompt_start> <music_prompt_end> <motion_prompt_start> <motion_prompt_end>'
            ] * batch_size

        duration = sequence_length / self.feature_frame_rate

        attributes = self._prepare_tokens_and_attributes(text_description, mode=mode)

        music_gen, motion_gen = self._generate_tokens(
            attributes, mode=mode, duration=duration, music_code=music_code, motion_code=motion_code,
            temperature=temperature, conditional_guidance_scale=conditional_guidance_scale
        )
        if music_code is None and motion_code is not None:
            return music_gen
        else:
            return motion_gen

    def generate_captions(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        return_caption_only: bool = False
    ) -> tp.Union[tp.List[str], tp.Tuple[tp.List[str], torch.LongTensor, torch.LongTensor]]:
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]
        batch_size = len(text_cond)
        descriptions: tp.List[str] = [
            '<music_prompt_start> <music_prompt_end> <motion_prompt_start> <motion_prompt_end>'
        ] * batch_size
        null_text_condition = self.prepare_text_condition(descriptions, mode='music_motion')  # use null condition

        music_motion_context = self.model.get_music_motion_context(
            music_code, motion_code, [], condition_tensors=null_text_condition
        )
        captions = self.text_model.generate_caption(music_motion_context)

        if return_caption_only:
            return captions
        else:
            return captions, music_code, motion_code

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
        self,
        descriptions: tp.Sequence[tp.Optional[str]],
        mode: str
    ) -> tp.List[ConditioningAttributes]:
        if mode == 'music2motion':
            dropped_out_descriptions = []
            # remove music descriptions
            for desc in descriptions:
                motion_description = desc.split('<music_prompt_end> ')[-1]
                dropped_out_descriptions.append(
                    '<music_prompt_start> <music_prompt_end> ' + motion_description
                )
            descriptions = dropped_out_descriptions
        elif mode == 'motion2music':
            dropped_out_descriptions = []
            # remove motion descriptions
            for desc in descriptions:
                music_description = desc.split(' <motion_prompt_start>')[0]
                dropped_out_descriptions.append(
                    music_description + ' <motion_prompt_start> <motion_prompt_end>'
                )
            descriptions = dropped_out_descriptions
        else:
            assert mode == 'music_motion'

        attributes = [ConditioningAttributes(text={'description': description}) for description in descriptions]

        # print debug info:
        for i in range(len(attributes)):
            print(f"Generating in {mode} with prompt {attributes[i].text['description']}")

        for attr in attributes:
            attr.wav['self_wav'] = WavCondition(
                torch.zeros((1, 1, 1), device=self.device),
                torch.tensor([0], device=self.device),
                sample_rate=[self.sample_rate],
                path=[None])

        return attributes

    def _generate_tokens(
        self,
        attributes: tp.List[ConditioningAttributes],
        mode: str,
        music_code: tp.Optional[torch.LongTensor] = None,
        motion_code: tp.Optional[torch.LongTensor] = None,
        duration: tp.Optional[float] = None,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: float = 1.
    ) -> tp.Tuple[torch.LongTensor, torch.LongTensor]:
        duration = self.duration if duration is None else duration
        total_gen_len = int(duration * self.feature_frame_rate)
        assert mode in ['music_motion', 'music2motion', 'motion2music']

        # generate by sampling from LM
        gen_tokens = self.model.generate(
            attributes,
            mode,
            music_code=music_code,
            motion_code=motion_code,
            max_gen_len=total_gen_len,
            use_sampling=self.generation_params['use_sampling'],
            temp=self.generation_params['temp'] if temperature is None else temperature,
            top_k=self.generation_params['top_k'],
            top_p=self.generation_params['top_p'],
            cfg_coef=self.generation_params['cfg_coef'] if conditional_guidance_scale is None else conditional_guidance_scale,
        )

        return gen_tokens

    def configure_optimizers(self):
        self.print_trainable_parameters()
        trainable_parameters = [p for p in self.parameters() if p.requires_grad]

        opt = torch.optim.AdamW(
            params=trainable_parameters,
            lr=self.optimization_config['learning_rate'],
            betas=self.optimization_config['betas'],
            weight_decay=self.optimization_config['weight_decay'],
            eps=self.optimization_config['eps']
        )

        if self.scheduler_config is None:
            return opt

        scheduler = instantiate_from_config(self.scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [
            {
                'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]

        return [opt], scheduler
