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
from transformers import CLIPModel, CLIPProcessor


logger = logging.getLogger(__name__)

class FrameTextAlignment:
    def __init__(
        self,
        device: torch.device,
    ):
        pretrained_model_name = 'openai/clip-vit-large-patch14'
        self.device = device
        logger.debug(f"Loding model {pretrained_model_name}")
        self.preprocessor = CLIPProcessor.from_pretrained(pretrained_model_name)
        self.model = CLIPModel.from_pretrained(pretrained_model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"Model {self.model.__class__.__name__} loaded")



    def preprocess(self, video, target_prompt):
        text_inputs = self.preprocessor(
            text=target_prompt, padding=True, truncation=True, max_length=77, return_tensors='pt').to(self.device)
        
        image_inputs = []
        for frame in video:
            image_inputs.append(self.preprocessor(
                images=frame, padding=True, truncation=True, max_length=77, return_tensors='pt').to(self.device))
        
        return text_inputs, image_inputs

    @torch.no_grad()
    def evaluate(self, video, target_prompt) -> float:
        text_inputs, image_inputs = self.preprocess(video, target_prompt)
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = []
        for image_input in image_inputs:
            image_embs = self.model.get_image_features(**image_input)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            score = (self.model.logit_scale.exp() * (text_embs @ image_embs.T)).cpu().squeeze().item()
            scores.append(score)

        return sum(scores) / len(scores)
