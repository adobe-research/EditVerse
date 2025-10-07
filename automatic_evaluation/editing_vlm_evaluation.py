# This file is based on code from https://github.com/wenhao728/awesome-diffusion-v2v Copyright (c) Wenhao, originally licensed under the MIT License. Modifications by Adobe Inc. are licensed under the Adobe Research License, Copyright 2025 Adobe Inc.
 
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import cv2
from openai import OpenAI
import numpy as np
import base64

logger = logging.getLogger(__name__)

class EditingVLMEvaluation:
    def __init__(
        self,
        gpt_api_key
    ):
        self.openai_client = OpenAI(api_key=gpt_api_key)
        self.sample_frames = 3

    def get_base64(self,frame):
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")
   
    @torch.no_grad()
    def evaluate(self, source_video, target_video, editing_prompt) -> float:
        scores = [] 
        for frame_idx in range(0, len(target_video), len(target_video)//self.sample_frames):
            while True:
                try:
                    before_frame = self.get_base64(np.array(source_video[frame_idx]))
                    after_frame = self.get_base64(np.array(target_video[frame_idx]))
                    response = self.openai_client.responses.create(
                        temperature=0,
                        model="gpt-4o-2024-11-20",
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": (
f'You are a meticulous video editing quality evaluator. Your task is to provide a detailed assessment of a video edit by comparing the original image with the edited image based on a given text prompt.\n\
Editing Prompt:\n{editing_prompt}\n\
Instructions:\n\
Analyze the provided image (the edited video frame) and evaluate how well the "Editing Prompt" has been executed. You will evaluate the edit across three distinct criteria. For each criterion, provide a score from 0 (worst) to 3 (best) and a brief justification. Finally, provide the total score.\n\
Your evaluation should focus on three key aspects:\n\
1. Prompt Following (Score: 0-3) \n\
Question: Does the edit accurately and completely fulfill the instructions in the "Editing Prompt"? \n\
Scoring Guide:\n\
- 3: The prompt is perfectly and completely followed.\n\
- 2: The prompt is mostly followed but with minor inaccuracies or omissions.\n\
- 1: The prompt is poorly followed or only partially executed.\n\
- 0: The prompt is completely ignored or the opposite was done. \n\
2. Edit Quality (Score: 0-3) \n\
Question: How is the visual quality of the edited area itself? Is it realistic, seamless, and free of artifacts (e.g., blurriness, distortion, unnatural textures)?\n\
Scoring Guide:\n\
- 3: The edit is of high visual quality, seamless, and artifact-free.\n\
- 2: The edit is good but has minor, noticeable artifacts.\n\
- 1: The edit is of low quality with significant, distracting artifacts.\n\
- 0: The edited area is extremely poor, garbled, or has completely failed.\n\
3. Background Consistency (Score: 0-3) \n\
Question: Have the areas that should not have been edited remained unchanged between the "Before" and "After" images? \n\
Scoring Guide:\n\
- 3: The areas that should not have been edited are perfectly preserved and stable. \n\
- 2: There are minor, subtle, but noticeable changes or flickers in the areas that should not have been edited.\n\
- 1: There are significant and distracting changes in the areas that should not have been edited. \n\
- 0: The areas that should not have been edited is completely or catastrophically altered. \n\
Please provide your evaluation in the following format: \n\
Prompt Following: [Your score, 0-3] - [Brief justification for the score.]\n\
Edit Quality: [Your score, 0-3] - [Brief justification for the score.]\n\
Background Consistency: [Your score, 0-3] - [Brief justification for the score.]\n\
Total Score: [Sum of the three scores]\n\
'
                                        )
                                    },
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/jpeg;base64,{before_frame}"
                                    },
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/jpeg;base64,{after_frame}"
                                    }
                                ]
                            }
                        ],
                    )

                    response = self.openai_client.responses.create(
                        temperature=0,
                        model="gpt-4o-2024-11-20",
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": (
                                        f'Please output the overall score mentioned in this sentence. Only output the overall score number. Sentence: {response.output_text}'
                                        )
                                    },
                                ]
                            }
                        ],
                    )
                    if response.output_text not in ['0','1','2','3','4','5','6','7','8','9']:
                        continue
                    scores.append(int(response.output_text))
                    break
                except Exception as e:
                    print(f"encounter error: {e}")

    
        return sum(scores) / len(scores)
