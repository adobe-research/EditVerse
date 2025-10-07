# This file is based on code from https://github.com/wenhao728/awesome-diffusion-v2v Copyright (c) Wenhao, originally licensed under the MIT License. Modifications by Adobe Inc. are licensed under the Adobe Research License, Copyright 2025 Adobe Inc.
 
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import warnings
from typing import Dict, List, Optional, Union
import os
import pandas as pd
import json
import torch
import decord
from tqdm import tqdm
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class EvaluatorWrapper:
    _temporal_consistency = ['clip_temporal_consistency', 'dino_temporal_consistency']
    _text_alignment = ['frame_text_alignment', 'video_text_alignment']
    _video_quality = ['pick_score_video_quality']
    _vlm_evaluation = ['editing_vlm_evaluation',]

    all_metrics = _temporal_consistency + _text_alignment + _video_quality + _vlm_evaluation

    def __init__(
        self,
        metrics: Union[str, List[str]] = 'all',
        test_json_path: str = "benchmark/test.json",
        gpt_api_key: str = "",
        device: Optional[torch.device] = None,
    ):
        self.metrics = metrics
        self._check_metrics()

        with open(test_json_path,"r") as f:
            self.test_json = json.load(f)

        self.gpt_api_key = gpt_api_key

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._initialize_models()

    def _check_metrics(self):
        if isinstance(self.metrics, str):
            if self.metrics == 'all':
                self.metrics = self.all_metrics
            else:
                self.metrics = self.metrics.split(',')
        not_supported_metrics = set(self.metrics) - set(self.all_metrics)
        self.metrics = [metric for metric in self.metrics if metric not in not_supported_metrics]
        if not_supported_metrics:
            warnings.warn(f'Unsupported metrics: {not_supported_metrics}')
        if self.metrics:
            logger.info(f'*******************************************')
            logger.info(f'Evaluating metrics: {self.metrics}')
            logger.info(f'*******************************************')
        else:
            raise ValueError('No supported metrics provided')

    def _initialize_models(self):
        if 'clip_temporal_consistency' in self.metrics:
            from automatic_evaluation.clip_temporal_consistency import ClipTemporalConsistency
            self.clip_temporal_consistency_evaluator = ClipTemporalConsistency(self.device)
        if 'dino_temporal_consistency' in self.metrics:
            from automatic_evaluation.dino_temporal_consistency import DinoTemporalConsistency
            self.dino_temporal_consistency_evaluator = DinoTemporalConsistency(self.device)
        if 'frame_text_alignment' in self.metrics:
            from automatic_evaluation.frame_text_alignment import FrameTextAlignment
            self.frame_text_alignment_evaluator = FrameTextAlignment(self.device)
        if 'video_text_alignment' in self.metrics:
            from automatic_evaluation.video_text_alignment import VideoTextAlignment
            self.video_text_alignment_evaluator = VideoTextAlignment(self.device)
        if 'pick_score_video_quality' in self.metrics:
            from automatic_evaluation.pick_score_video_quality import PickScoreVideoQuality
            self.pick_score_video_quality_evaluator = PickScoreVideoQuality(self.device)
        if 'editing_vlm_evaluation' in self.metrics:
            from automatic_evaluation.editing_vlm_evaluation import EditingVLMEvaluation
            self.editing_vlm_evaluation_evaluator = EditingVLMEvaluation(self.gpt_api_key)


    def evaluate(
        self,
        generate_results_dir: str = "results/EditVerse_original",
        output_csv: str = "EditVerse_original.csv",
    ) -> Dict[str, Dict[str, List]]:
        
        all_evaluation_results_df = pd.DataFrame(columns=["id"]+self.metrics+["type"])

        for item_key, item_value in tqdm(self.test_json.items()):
            item_evaluation_results = [item_key]
            
            generated_video = os.path.join(generate_results_dir,item_key,"generate.mp4")
            video_reader = decord.VideoReader(generated_video, num_threads=1)
            video = []    
            for i in range(len(video_reader)):
                video.append(Image.fromarray(video_reader[i].asnumpy()))

            original_video = os.path.join(generate_results_dir,item_key,"video1.mp4")
            video_reader = decord.VideoReader(original_video, num_threads=1)
            source_video = []    
            for i in range(len(video_reader)):
                source_video.append(Image.fromarray(video_reader[i].asnumpy()))

            editing_prompt = item_value["<text>"]

            target_prompt = item_value["target_prompt"]

            for metric in self.metrics:
                logger.info(f'evaluating {metric}')
                if metric == 'clip_temporal_consistency':
                    item_evaluation_results.append(self.clip_temporal_consistency_evaluator.evaluate(video))
                elif metric == 'dino_temporal_consistency':
                    item_evaluation_results.append(self.dino_temporal_consistency_evaluator.evaluate(video))
                elif metric == 'frame_text_alignment':
                    item_evaluation_results.append(self.frame_text_alignment_evaluator.evaluate(video, target_prompt))
                elif metric == 'video_text_alignment':
                    item_evaluation_results.append(self.video_text_alignment_evaluator.evaluate(video, target_prompt))
                elif metric == 'pick_score_video_quality':
                    item_evaluation_results.append(self.pick_score_video_quality_evaluator.evaluate(video, target_prompt))
                elif metric == "editing_vlm_evaluation":
                    item_evaluation_results.append(self.editing_vlm_evaluation_evaluator.evaluate(source_video, video, editing_prompt))


            item_evaluation_results +=[item_value["type"]]
            
            all_evaluation_results_df.loc[len(all_evaluation_results_df)]=item_evaluation_results

            all_evaluation_results_df.to_csv(output_csv, index=False)

        logger.info(f"Finish evaluation, saving results to {output_csv} ...")
        numeric_cols = all_evaluation_results_df.select_dtypes(include=np.number)
        averages = numeric_cols.mean()
        all_evaluation_results_df.loc['Average'] = pd.Series(averages, name='Average')
        all_evaluation_results_df.loc['Average', 'id'] = 'Average'
        all_evaluation_results_df.to_csv(output_csv, index=False)
        logger.info(f"Finish results saving")

        return all_evaluation_results_df
