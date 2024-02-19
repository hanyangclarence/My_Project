import torch
import torch.nn as nn
from typing import Sequence, List

from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class TextGenerator(nn.Module):
    def __init__(self, model: str = "t5-base", max_length: int = 64, context_dim: int = 1024, self_dim: int = 768):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.add_tokens('<separation>')
        self.max_length = max_length
        self.context_dim = context_dim
        self.self_dim = self_dim

        self.model = T5ForConditionalGeneration.from_pretrained(model)

        self.context_proj = nn.Linear(context_dim, self_dim)

    def forward(self, texts: Sequence[str], music_motion_context: torch.Tensor, mode: str) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )

        device = next(self.model.parameters()).device
        labels = encoded["input_ids"].to(device)
        decoder_attention_mask = encoded["attention_mask"].to(device)

        if labels.shape[-1] > 300:
            print('!!!!!!!!', texts)

        music_motion_context = self.context_proj(music_motion_context)
        music_motion_context = BaseModelOutput(music_motion_context)

        labels[labels == 0] = -100

        loss = self.model.forward(
            encoder_outputs=music_motion_context,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        ).loss

        return loss

    def generate_caption(self, music_motion_context: torch.Tensor, mode: str) -> List[str]:
        music_motion_context = self.context_proj(music_motion_context)
        music_motion_context = BaseModelOutput(music_motion_context)

        outputs = self.model.generate(
            encoder_outputs=music_motion_context,
            do_sample=False,
            max_length=256,
            num_beams=1
        )

        captions = []
        for output in outputs:
            captions.append(self.tokenizer.decode(output, skip_special_tokens=True))

        return captions
