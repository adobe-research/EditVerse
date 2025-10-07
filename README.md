# EditVerse


This repository contains the **instruction-based video editing evaluation code for EditVerseBench** in paper "EditVerse: A Unified Framework for Editing and Generation via In-Context Learning".


> [Xuan Ju](https://juxuan27.github.io/)<sup>12</sup>, [Tianyu Wang](https://scholar.google.com/citations?user=yRwZIN8AAAAJ&hl=zh-CN)<sup>1</sup>, [Yuqian Zhou](https://yzhouas.github.io/)<sup>1</sup>, [He Zhang](https://sites.google.com/site/hezhangsprinter)<sup>1</sup>, [Qing Liu](https://qliu24.github.io/)<sup>1</sup>, [Nanxuan Zhao](https://www.nxzhao.com/)<sup>1</sup>, [Zhifei Zhang](https://zzutk.github.io/)<sup>1</sup>, [Yijun Li](https://yijunmaverick.github.io/)<sup>1</sup>, [Yuanhao Cai](https://caiyuanhao1998.github.io/)<sup>3</sup>, [Shaoteng Liu](https://www.shaotengliu.com/)<sup>1</sup>, [Daniil Pakhomov](https://scholar.google.com/citations?user=UI10l34AAAAJ&hl=en)<sup>1</sup>, [Zhe Lin](https://sites.google.com/site/zhelin625/)<sup>1</sup>, [Soo Ye Kim](https://sites.google.com/view/sooyekim)<sup>1*</sup>, [Qiang Xu](https://cure-lab.github.io/)<sup>2*</sup><br>
> <sup>1</sup>Adobe Research <sup>2</sup>The Chinese University of Hong Kong <sup>3</sup>Johns Hopkins University <sup>*</sup>Corresponding Author



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

You can download the source videos from Pixabay using the links in `EditVerseBench/EditVerseBench/test.json`. The benchmark file structure should be like:

```
EditVerseBench/EditVerseBench/
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

## License
Files under `./automatic_evaluation/viclip` are from [InternVideo](https://github.com/OpenGVLab/InternVideo) and under [Apache 2.0 License](https://github.com/OpenGVLab/InternVideo?tab=Apache-2.0-1-ov-file#readme). Files under `./automatic_evaluation` except for those under the folder `viclip` are modified from [awesome-diffusion-v2v](https://github.com/wenhao728/awesome-diffusion-v2v/tree/main) under [MIT License](https://github.com/wenhao728/awesome-diffusion-v2v/tree/main?tab=MIT-1-ov-file#readme) and modifications by Adobe are under [Adobe Research License](https://github.com/OneAdobe/EditVerse/blob/main/LICENSE.md). All other materials are licensed under [Adobe Research License](https://github.com/OneAdobe/EditVerse/blob/main/LICENSE.md).
