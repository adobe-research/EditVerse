# This file is based on code from https://github.com/wenhao728/awesome-diffusion-v2v Copyright (c) Wenhao, originally licensed under the MIT License. Modifications by Adobe Inc. are licensed under the Adobe Research License, Copyright 2025 Adobe Inc.
 
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

import torch

from automatic_evaluation.viclip import SimpleTokenizer, ViCLIP, clip_image_transform

logger = logging.getLogger(__name__)

class VideoTextAlignment:

    def __init__(
        self,
        device: torch.device,
        pretrained_tokenizer: str = None,
        pretrained_checkpoint: str = None,
    ):
        pretrained_tokenizer = "automatic_evaluation/ckpt/bpe_simple_vocab_16e6.txt.gz"
        pretrained_checkpoint = "automatic_evaluation/ckpt/ViClip-InternVid-10M-FLT.pth"

        if not os.path.exists(pretrained_tokenizer):
            os.system(f"wget https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz -P automatic_evaluation/ckpt")
        if not os.path.exists(pretrained_checkpoint):
            os.system(f"wget https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth -P automatic_evaluation/ckpt")

        self.device = device
        logger.debug(f"Loding model {pretrained_checkpoint}")
        tokenizer = SimpleTokenizer(bpe_path=pretrained_tokenizer)
        self.model = ViCLIP(tokenizer=tokenizer, pretrain=pretrained_checkpoint)
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"Model {self.model.__class__.__name__} loaded")

        self.image_transform = clip_image_transform(224)


    def preprocess(self, video, target_prompt):
        step = len(video) / 8
        video = [video[int(i * step)] for i in range(8)]
        text_inputs = target_prompt
        
        frames = []
        for frame in video:
            frames.append(self.image_transform(frame))
        video_inputs = torch.stack(frames).to(self.device)[None] # (1, T, C, H, W)
        logger.debug(f"video_inputs shape: {video_inputs.shape}")

        return text_inputs, video_inputs

    @torch.no_grad()
    def evaluate(self, video, target_prompt) -> float:
        text_inputs, video_inputs = self.preprocess(video, target_prompt)

        text_embs: torch.Tensor = self.model.encode_text(text_inputs).float()
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        logger.debug(f"text_embs shape: {text_embs.shape}")

        video_embs: torch.Tensor = self.model.encode_vision(video_inputs, test=True).float()
        video_embs = video_embs / torch.norm(video_embs, dim=-1, keepdim=True)
        logger.debug(f"video_embs shape: {video_embs.shape}")

        score = (text_embs @ video_embs.T).cpu().squeeze().item()

        return score*100
