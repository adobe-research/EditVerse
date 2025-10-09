# EditVerse


This repository contains the **instruction-based video editing evaluation code for EditVerseBench** in the paper "EditVerse: A Unified Framework for Editing and Generation via In-Context Learning".


> [Xuan Ju](https://juxuan27.github.io/)<sup>12</sup>, [Tianyu Wang](https://scholar.google.com/citations?user=yRwZIN8AAAAJ&hl=zh-CN)<sup>1</sup>, [Yuqian Zhou](https://yzhouas.github.io/)<sup>1</sup>, [He Zhang](https://sites.google.com/site/hezhangsprinter)<sup>1</sup>, [Qing Liu](https://qliu24.github.io/)<sup>1</sup>, [Nanxuan Zhao](https://www.nxzhao.com/)<sup>1</sup>, [Zhifei Zhang](https://zzutk.github.io/)<sup>1</sup>, [Yijun Li](https://yijunmaverick.github.io/)<sup>1</sup>, [Yuanhao Cai](https://caiyuanhao1998.github.io/)<sup>3</sup>, [Shaoteng Liu](https://www.shaotengliu.com/)<sup>1</sup>, [Daniil Pakhomov](https://scholar.google.com/citations?user=UI10l34AAAAJ&hl=en)<sup>1</sup>, [Zhe Lin](https://sites.google.com/site/zhelin625/)<sup>1</sup>, [Soo Ye Kim](https://sites.google.com/view/sooyekim)<sup>1*</sup>, [Qiang Xu](https://cure-lab.github.io/)<sup>2*</sup><br>
> <sup>1</sup>Adobe Research <sup>2</sup>The Chinese University of Hong Kong <sup>3</sup>Johns Hopkins University <sup>*</sup>Corresponding Author


<p align="center">
  <a href="http://editverse.s3-website-us-east-1.amazonaws.com/">🌐 Project Page</a> |
  <a href="https://arxiv.org/abs/2509.20360">📜 Arxiv</a> |
  <a href="https://huggingface.co/datasets/sooyek/EditVerseBench">🤗 Benchmark</a> |
  <a href="https://docs.google.com/presentation/d/1dBg3lZDFa8mRRIrOVEU_xDgzedufbwzr/edit?usp=sharing&ouid=100286465794673637256&rtpof=true&sd=true">📹 Slides</a> |
  <a href="http://editverse.s3-website-us-east-1.amazonaws.com/comparison.html">👀 Comparison</a>
</p>


## Setup Environment

**(Optional) Create a Conda environment**

```
conda create -n EditVerse python=3.10
conda activate EditVerse
```

**Install Pytorch** 

(You may adjust the version or CUDA support depending on your hardware)

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**Install required packages**

```
pip install -r requirements.txt
```


## Download Benchmark & Results

**Download benchmark dataset**

```
git lfs install
git clone https://huggingface.co/datasets/sooyek/EditVerseBench
```

**Download the videos**

The source videos cannot be directly distributed due to licensing restrictions. Instead, you can download them using the provided script with the Pixabay API. (The network connection may occasionally fail, so you might need to run the script multiple times.) 

> ⚠️ Note: Please remember to revise the API key to your own key in download_source_video.py. You can find the API key [here](https://pixabay.com/api/docs/#api_search_images) (marked in Parameters-key(required) on the website). The API is free, but you need to sign up for an account to get the API key. 

```
cd EditVerseBench
python download_source_video.py
```


The benchmark file structure should be like:

```
EditVerseBench/
  ├── test.json
  ├── depths/
  │   ├── xx.mp4
  ├── edited_first_frame/
  │   ├── xx.mp4
  ├── images/
  │   ├── xx.mp4
  ├── inpaint_video_and_masks/
  │   ├── xx.mp4
  ├── poses/
  │   ├── xx.mp4
  ├── sketchs/
  │   ├── xx.mp4
  ├── videos/
  │   ├── xx.mp4
```

**Unpack comparison results**

```
cd EditVerseBench
tar -zxvf EditVerse_Comparison_Results.tar.gz
rm EditVerse_Comparison_Results.tar.gz
```


## Evaluation

**Command**

```
python eval.py --metrics [metrics] \
--test_json_path EditVerseBench/EditVerseBench/test.json \
--generate_results_dir [results_dir] \
--output_csv [output_csv] \
--gpt_api_key [your_api_key]
```

**Arguments**

- `metrics`: Use all to evaluate all metrics.
    
    To select specific metrics, provide a comma-separated list (no spaces). Example: `clip_temporal_consistency,dino_temporal_consistency`

    Supported metrics include:
    - clip_temporal_consistency
    - dino_temporal_consistency
    - frame_text_alignment
    - video_text_alignment
    - pick_score_video_quality
    - editing_vlm_evaluation

- `test_json_path`: Path to the benchmark entrypoint JSON file.
- `generate_results_dir`: Directory containing generated results (must follow the required structure).
- `output_csv`: Path to save the evaluation CSV file.
- `gpt_api_key`: penAI API key (required for editing_vlm_evaluation).


**Example**

Evaluate the provided EditVerse results and save output to EditVerse_eval.csv:

```
python eval.py --metrics all \
--test_json_path EditVerseBench/EditVerseBench/test.json \
--generate_results_dir EditVerseBench/EditVerse_Comparison_Results/EditVerse \
--output_csv EditVerse_eval.csv \
--gpt_api_key [Your API key]
```

👉 Pre-computed evaluation results for EditVerse and previous methods are available at: `EditVerseBench/automatic_evaluation_results`.



## Evaluate Your Own Model


You can also evaluate your model outputs by following the same format.

**Step 1: Refer to benchmark JSON format**

See `EditVerseBench/EditVerseBench/test.json` for reference.

Each entry looks like this:

```
{
    "0": {
        "<text>": "<video1> Add a small golden crown ...",
        "<video1>": "videos/174008-850361316.mp4",
        "<video1> link": "https://pixabay.com/videos/woman-smile-communication-gesture-174008/",
        "direction": "horizontal",
        "target_prompt": "A young woman stands outside in front of ...",
        "type": "add object",
        "source_prompt": "A young woman stands outside in front of ..."
    },
    "1": {
        ...
    },
    ...
}
```
Key fields:
- `<text>`: A natural language instruction describing the required edit in an interleaved format.
  - The instruction may include special tags such as `<video1>`, `<video2>`, or `<image1>`.
  - Each tag corresponds to a specific key field defined in the same JSON entry.
- `<video1>`: The local file path of the source video.
- `<video1> link`: The reference URL pointing to the source video’s original location.
- `direction`: horizontal or vertical.
- `target_prompt`: A detailed textual description of the desired edited video outcome.
- `type`: The category of the edit
- `source_prompt`: A description of the original, unedited video.

**Step 2: Format your results**

After generating results with your model, arrange files as follows:

```
Your_Folder/
  ├── 0/
  │   ├── generate.mp4   # model-generated video
  │   └── video1.mp4     # source video
  ├── 1/
  │   ├── generate.mp4
  │   └── video1.mp4
  ...
```

**Step 3: Run evaluation**

```
python eval.py --metrics all \
--test_json_path EditVerseBench/EditVerseBench/test.json \
--generate_results_dir [Your_Folder] \
--output_csv [Your_Results.csv] \
--gpt_api_key [your_api_key]

```


## Benchmark Results
<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="1">VLM evaluation</th>
      <th colspan="1">Video Quality</th>
      <th colspan="2">Text Alignment</th>
      <th colspan="2">Temporal Consistency</th>
    </tr>
    <tr>
      <th>Editing Quality ↑</th>
      <th>Pick Score ↑</th>
      <th>Frame ↑</th>
      <th>Video ↑</th>
      <th>CLIP ↑</th>
      <th>DINO ↑</th>
    </tr>
  </thead>
  <tbody>
    <!-- Attention Manipulation -->
    <tr>
      <td colspan="7" style="text-align:center; font-weight:bold;">Attention Manipulation (Training-free)</td>
    </tr>
    <tr>
      <td><b>TokenFlow</b></td>
      <td>5.26</td><td>19.73</td><td>25.57</td><td>22.70</td><td>98.36</td><td>98.09</td>
    </tr>
    <tr>
      <td><b>STDF</b></td>
      <td>4.41</td><td>19.45</td><td>25.24</td><td>22.26</td><td>96.04</td><td>95.22</td>
    </tr>
    <!-- First-Frame Propagation -->
    <tr>
      <td colspan="7" style="text-align:center; font-weight:bold;">First-Frame Propagation (w/ End-to-End Training)</td>
    </tr>
    <tr>
      <td><b>Señorita-2M</b></td>
      <td>6.97</td><td>19.71</td><td>26.34</td><td>23.24</td><td>98.05</td><td>97.99</td>
    </tr>
    <!-- Instruction-Guided -->
    <tr>
      <td colspan="7" style="text-align:center; font-weight:bold;">Instruction-Guided (w/ End-to-End Training)</td>
    </tr>
    <tr>
      <td><b>InsV2V</b></td>
      <td>5.21</td><td>19.39</td><td>24.99</td><td>22.54</td><td>97.15</td><td>96.57</td>
    </tr>
    <tr>
      <td><b>Lucy Edit</b></td>
      <td>5.89</td><td>19.67</td><td>26.00</td><td>23.11</td><td>98.49</td><td>98.38</td>
    </tr>
    <tr>
      <td><b>Ours (Ours)</b></td>
      <td><b>7.65</b></td><td><b>20.07</b></td><td><b>26.73</b></td><td><b>23.93</b></td><td><b>98.56</b></td><td><b>98.42</b></td>
    </tr>
    <!-- Closed-Source -->
    <!-- <tr>
      <td colspan="7" style="text-align:center; font-weight:bold; color:gray;">Closed-Source Commercial Models</td>
    </tr>
    <tr style="color:gray;">
      <td>Runway Aleph</td>
      <td>7.44</td><td>20.42</td><td>27.70</td><td>24.27</td><td>98.94</td><td>98.60</td>
    </tr> -->
  </tbody>
</table>


## License
Files under `./automatic_evaluation/viclip` are from [InternVideo](https://github.com/OpenGVLab/InternVideo) and under [Apache 2.0 License](https://github.com/OpenGVLab/InternVideo?tab=Apache-2.0-1-ov-file#readme). Files under `./automatic_evaluation` except for those under the folder `viclip` are modified from [awesome-diffusion-v2v](https://github.com/wenhao728/awesome-diffusion-v2v/tree/main) under [MIT License](https://github.com/wenhao728/awesome-diffusion-v2v/tree/main?tab=MIT-1-ov-file#readme) and modifications by Adobe are under [Adobe Research License](https://github.com/OneAdobe/EditVerse/blob/main/LICENSE.md). All other materials are licensed under [Adobe Research License](https://github.com/OneAdobe/EditVerse/blob/main/LICENSE.md).

## Cite Us

If you find our work useful for your research, please consider citing our paper:

```
@article{ju2025editverse,
  title   = {EditVerse: Unifying Image and Video Editing and Generation with In-Context Learning},
  author  = {Xuan Ju and Tianyu Wang and Yuqian Zhou and He Zhang and Qing Liu and Nanxuan Zhao and Zhifei Zhang and Yijun Li and Yuanhao Cai and Shaoteng Liu and Daniil Pakhomov and Zhe Lin and Soo Ye Kim and Qiang Xu},
  journal = {arXiv preprint arXiv:2509.20360},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.20360}
}
```
