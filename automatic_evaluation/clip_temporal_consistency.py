# This file is based on code from https://github.com/wenhao728/awesome-diffusion-v2v Copyright (c) Wenhao, originally licensed under the MIT License. Modifications by Adobe Inc. are licensed under the Adobe Research License, Copyright 2025 Adobe Inc.
 
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
@Ref     :   
    https://github.com/openai/CLIP
    https://github.com/mlfoundations/open_clip
    https://huggingface.co/docs/transformers/model_doc/clip#clip
'''
import logging
from pathlib import Path

import torch
from transformers import CLIPImageProcessor, CLIPVisionModel


logger = logging.getLogger(__name__)

class ClipTemporalConsistency:
    def __init__(
        self,
        device: torch.device,
    ):
        pretrained_model_name = 'openai/clip-vit-large-patch14'
        self.device = device
        logger.debug(f"Loding model {pretrained_model_name}")
        self.preprocessor = CLIPImageProcessor.from_pretrained(pretrained_model_name)
        self.model = CLIPVisionModel.from_pretrained(pretrained_model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"Model {self.model.__class__.__name__} loaded")

    def preprocess(self, video):
        frames = []
        for i, frame in enumerate(video):
            frames.append(self.preprocessor(frame, return_tensors='pt').pixel_values)
        return frames

    @torch.no_grad()
    def evaluate(self, video) -> float:
        frames = self.preprocess(video)

        similarity = []
        former_feature = None
        for i, frame in enumerate(frames):
            frame = frame.to(self.device)
            feature: torch.Tensor = self.model(pixel_values=frame).pooler_output
            feature = feature / torch.norm(feature, dim=-1, keepdim=True)

            if i > 0:
                sim = max(0, (feature @ former_feature.T).cpu().squeeze().item())
                similarity.append(sim)
            former_feature = feature
        return sum(similarity) / len(similarity)
