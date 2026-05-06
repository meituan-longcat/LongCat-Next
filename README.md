# LongCat-Next

<div align="center">
  <img src="https://raw.githubusercontent.com/meituan-longcat/LongCat-Flash-Chat/main/figures/longcat_logo.svg"
       width="300"
       alt="LongCat Logo"/>
</div>

<hr>

<div align="center" style="line-height: 1;">
    <a href="https://longcat.chat/longcat-next/intro" target="_blank" style="margin: 2px;">
        <img alt="Blog" src="https://img.shields.io/badge/Blog-LongCatNext-white?logo=safari&logoColor=white&color=purple" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://huggingface.co/meituan-longcat/LongCat-Next" target="_blank" style="margin: 2px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCatNext-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://github.com/meituan-longcat/LongCat-Next" target="_blank" style="margin: 2px;">
        <img alt="GitHub" src="https://img.shields.io/badge/GitHub-LongCatNext-white?logo=github&logoColor=white&color=a4b5d5" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://longcat.chat/longcat-next" target="_blank" style="margin: 2px;">
        <img alt="Demo" src="https://img.shields.io/badge/Demo-LongCatNext-white?logo=googleplay&logoColor=white&color=eabcdd" style="display: inline-block; vertical-align: middle;"/>
    </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/EXsG52D8SW" target="_blank" style="margin: 2px;">
    <img src="https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white" />
  </a>
  <a href="https://github.com/meituan-longcat/LongCat-Flash-Chat/blob/main/figures/wechat_official_accounts.png" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-LongCat-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/Meituan_LongCat" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-LongCat-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2603.27538" target="_blank" style="margin: 2px;">
        <img alt="Paper" src="https://img.shields.io/badge/arXiv-2603.27538-b31b1b?logo=arxiv&logoColor=b31b1b" style="display: inline-block; vertical-align: middle;"/>  
    </a>
  <a href="https://huggingface.co/meituan-longcat/LongCat-Next/blob/main/LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<p align="center">
  <a href="https://github.com/meituan-longcat/LongCat-Next/blob/main/tech_report.pdf">
  <b>Tech Report</b>&nbsp;📄
  </a>
</p>





## Model Introduction

![evaluation](./assets/overview.jpg)


We develop **LongCat-Next**, a native multimodal model that processes text, vision, and audio under a single autoregressive objective with minimal inductive bias beyond the language paradigm. As an industrial-strength foundation model with A3B model size, it excels at seeing, creating, and talking, achieving strong performance across a wide range of multimodal benchmarks. In particular, leveraging semantically complete discrete representations, it surpasses the long-standing performance ceiling of discrete vision modeling on understanding tasks, and provides a unified solution for visual understanding and generation. This success demonstrates that discrete tokens can universally represent multimodal signals and be deeply internalized within a single discrete embedding space. We further provide extensive experiments to analyze this unified discrete training paradigm and uncover several interesting findings.

As a meaningful attempt toward native multimodality, we open-source the **LongCat-Next** and its tokenizers, hoping to foster further research and development in the community.


### Key Features

This work primarily addresses the fundamental barrier to native multimodality through a design philosophy that prioritizes simplicity, treating vision and audio as intrinsic extensions of language. As a step toward this goal, we present LongCat-Next, a discrete native multimodal model that achieves industrial-strength performance within discrete frameworks while remaining highly competitive across a wide range of specialized domains. Built upon the LongCat-Flash-Lite MoE backbone (A3B) as a _multi-task_ learner, the model unifies language, vision, and audio within a single discrete framework. In this paper, we make the following principal contributions:

#### 🌟  Discrete Native Autoregression Paradigm (DiNA).
We introduce DiNA, a unified paradigm that extends next-token prediction from language to native multimodality, which internalizes diverse modalities into a shared token space. It simplifies multimodal modeling by creating modality-aware tokenizer-detokenizer pairs and leveraging the established training infrastructure of large language models.


#### 🌟  Semantic Completeness for Discrete Visual Representation.
We improve discrete visual modeling by combining Semantic-and-Aligned Encoders (SAE) with Residual Vector Quantization (RVQ). This integration creates hierarchical discrete tokens that preserve both semantic abstraction and fine-grained visual details, surpassing traditional representation limitations.


#### 🌟  Discrete Native-Resolution Vision Transformer (dNaViT).
Analogous to linguistic tokenizers, we propose dNaViT as a highly flexible, unified discrete interface for vision that extracts semantic features as "visual words", constructing a hierarchical representation space supporting dynamic tokenization and detokenization. dNaViT integrates seamlessly with large language models, ensuring high performance without degradation.

#### 🌟  Exceling in Seeing, Creating, and Talking in a Unified Model.
Within the framework of DiNA, visual understanding and generation are elegantly reformulated as two manifestations of the same predictive process without performance compromise. This formulation bridges the long-standing architectural divide while introducing minimal interference between these traditionally competing objectives and preserving core language capabilities. Remarkably, LongCat-Next achieves competitive performance with specialized understanding models, while maintaining strong generative quality even under a 28× compression ratio, particularly in text rendering, while also excelling in advanced speech comprehension, low-latency voice conversation, and customizable voice cloning.


Please refer to our [technical report](./tech_report.pdf) for details!



## Evaluation Results

![evaluation](./assets/evaluation.png)




## Quick Start
To use LongCat-Next with transformers, we need at least 3 GPUs (80GB VRAM each, e.g., H100/A100 80GB), and we recommend the following environment:
* `python` >= 3.10
* `torch` >= 2.6
* `transformers` >= 4.57.6
* `accelerate` >= 1.10.0

```shell
# (Install python=3.10, ffffmpeg<7, soundfile==0.13.1)
conda env create -f environment.yml -v

# (Install torch and other pip dependencies)
pip install -r requirements.txt && pip install -r requirements-post.txt --no-build-isolation
```

Basic Usage Example:
- Remember to modify `WEIGHT_PATH_TO_LONGCAT_NEXT` in `./config.json`, because decoders use lazy loading.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Load model
model_name = "meituan-longcat/LongCat-Next"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, fix_mistral_regex=True)
model.text_tokenizer = tokenizer # Dynamic binding
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Set messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What book is this?<longcat_img_start>./assets/book.png<longcat_img_end>"}
]

# Apply chat-template
text_input = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(f"{text_input=}")

# Preprocessing
text_inputs, visual_inputs, audio_inputs = processor(text=text_input, return_tensors="pt")
text_inputs = text_inputs.to(model.device)
if visual_inputs is not None:
    visual_inputs = visual_inputs.to(model.device)
if audio_inputs is not None:
    audio_inputs = audio_inputs.to(model.device)

# AR
with torch.no_grad():
    outputs = model.generate(
        input_ids=text_inputs["input_ids"],
        visual_inputs=visual_inputs,
        audio_inputs=audio_inputs,
        return_dict_in_generate=True,
    )

# Text decoding
output_input_ids = outputs.sequences
text_output = tokenizer.decode(output_input_ids[0][len(text_inputs["input_ids"][0]):], skip_special_tokens=True)
print(f"{text_output=}")

# Images decoding
output_visual_ids = outputs.visual_ids
if output_visual_ids.size(0) > 0:
    image_path_list = model.model.decode_visual_ids_and_save(
        output_visual_ids,
        save_prefix="./output_image",
        **model.generation_config.visual_generation_config["custom_params"],
    )
    print(f"{image_path_list=}")

# Audio decoding
output_audio_text_ids = outputs.audio_text_ids
output_audio_ids = outputs.audio_ids
if output_audio_text_ids.size(-1) > 0:
    audio_text = tokenizer.decode(output_audio_text_ids[0], skip_special_tokens=True)
    print(f"{audio_text=}")
if output_audio_ids.size(0) > 0:
    audio_path_list = model.model.decode_audio_ids_and_save(
        output_audio_ids,
        save_prefix="./output_audio",
        **model.generation_config.audio_generation_config["custom_params"],
    )
    print(f"{audio_path_list=}")
```


<details>
<summary>Text - Tool Calling Example</summary>

```python
from parse_model_response import parse_model_response

tools = [
    {
        "type": "function",
        "function": {
            "name": "func_add",
            "description": "Calculate the sum of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {"type": "number", "description": "The first addend"},
                    "x2": {"type": "number", "description": "The second addend"}
                },
                "required": ["x1", "x2"]
            }
        }
    }
]
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Please tell me what is $$125679 + 234519$$?"},
    {
        "role": "assistant",
        "content": "I'll calculate the sum of 125679 and 234519 for you.",
        "tool_calls": [{"type": "function", "function": {"name": "func_add", "arguments": {"x1": 125679, "x2": 234519}}}]
    },
    {"role": "tool", "name": "func_add", "content": '{"ans": 360198}'}
]

text_input = tokenizer.apply_chat_template(
    messages,
    tools=tools, # add tools here
    tokenize=False,
    add_generation_prompt=True,
)
print(f"{text_input=}")


# Preprocessing - AR - Text decoding
...

# Results parsing
parsed_message = parse_model_response(text_output.strip("\n"), tools)
print(f"{parsed_message=}")
```
See [`parse_model_response.py`](./parse_model_response.py) for detailed implementation and examples.

</details>


<details>
<summary>Image - Understanding Example</summary>

```python
# Simply replace the messages in the main example with the messages below.
messages = [
    {"role": "user", "content": "What book is this?<longcat_img_start>./assets/book.png<longcat_img_end>"}
]
```

</details>


<details>
<summary>Image - Generation Example</summary>

```python
# Simply replace the messages in the main example with the messages below.
# Suffix user content with '<longcat_img_start>' to force image generation.
messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "A small kitten sitting naturally on a moss-covered forest floor, centered in the frame, holding a rectangular wooden sign gently with its front paws resting over the top edge. The kitten has soft, fluffy fur, a natural relaxed posture, and a calm, curious expression with a slightly open mouth (not exaggerated), looking directly at the camera.\n\nThe sign is positioned firmly in front of the kitten\'s chest, supported by its paws, with realistic contact and no floating effect. The board reads \"LongCat-Next: Lexicalizing Modalities as Discrete Tokens\" in clean, sharp black text, perfectly legible.\n\nThe environment is a lush forest with tall trees, ferns, and soft green foliage. The ground is covered with moss and small plants. Background softly blurred with natural depth of field. Lighting is soft, diffused sunlight filtering through the trees, creating gentle highlights and shadows. Realistic photography style, natural colors, high detail, no cartoonish exaggeration.<longcat_img_start>"}
]
```

</details>


<details>
<summary>Audio - Audio-to-Text Example</summary>

```python
# Simply replace the messages in the main example with the messages below.
messages = [
    {"role": "user", "content": "<longcat_audio_start>./assets/math1.wav<longcat_audio_end>"}
]

```

</details>

<details>
<summary>Audio - Audio-to-Audio Example</summary>

```python
# Simply replace the messages in the main example with the messages below.
# Suffix user content with '<longcat_audiogen_start>' to force audio generation.
messages = [
    {"role": "system", "content": "Replicate the voice in the audio clip to formulate an answer:<longcat_audio_start>./assets/system_audio.wav<longcat_audio_end>"},
    {"role": "user", "content": "<longcat_audio_start>./assets/math1.wav<longcat_audio_end><longcat_audiogen_start>"}
]
```

</details>

<details>
<summary>Audio - Speech Synthesis Example</summary>

```python
# Simply replace the messages in the main example with the messages below.
# Suffix user content with '<longcat_audiogen_start>' to force audio generation.
messages = [
    {"role": "system", "content": "Replicate the voice in the audio clip to formulate an answer:<longcat_audio_start>./assets/vc_zh3.wav<longcat_audio_end>"},
    {"role": "user", "content": "用这个声音合成以下内容：明天的meeting在三楼的Conference Room举行。<longcat_audiogen_start>"}
]
```

</details>


<!-- > [!Tip] -->

> We recommend using the following set of sampling parameters for generation:
> 
> - Text: `{"max_new_tokens":2048,"do_sample":false}`
> - Image - Understanding: `{"max_new_tokens":1024,"do_sample":true,"temperature":0.4,"top_k":40,"top_p":0.85,"repetition_penalty":1.1}`
> - Image - Generation: `{"max_new_tokens":2048,"do_sample":false,"visual_generation_config":{"do_sample":true,"temperature":0.5,"top_p":0.75,"top_k":1024,"custom_params":{"cfg_scale":3,"token_h":37,"token_w":37,"anyres_prefix":"<longcat_img_token_size>{h} {w}</longcat_img_token_size>"}}}`
> - Audio - Audio-to-Text: `{"max_new_tokens":1024,"do_sample":true,"temperature":0.2,"top_k":20,"top_p":0.85,"repetition_penalty":1.1}`
> - Audio - Audio-to-Audio/Speech Synthesis: `{"max_new_tokens":2048,"do_sample":true,"temperature":0.2,"top_k":20,"top_p":0.85,"repetition_penalty":1.1,"audio_generation_config":{"audio_parallel_decoding":false,"do_sample":true,"temperature":0.5,"top_k":5,"top_p":0.85,"repetition_penalty":1.3,"custom_params":{"sampling_rate":24000,"wave_concat_overlap":1200}}}`
>
> Please note that the support for sampling parameters varies according to inference frameworks(For transformers, the inference parameter configuration is located in `./generation_config.json`).



## Deployment

We have implemented basic adaptations in SGLang to support the deployment of LongCat-Next. Please refer to this repository for more information: [meituan-longcat/LongCat-Next-inference](https://github.com/meituan-longcat/LongCat-Next-inference)



## License Agreement
This repository, including both the model weights and the source code, is released under the **MIT License**.

Any contributions to this repository are licensed under the MIT License, unless otherwise stated. This license does not grant any rights to use Meituan trademarks or patents.

For details, see the [LICENSE](./LICENSE) file.

## Usage Considerations
This model has not been specifically designed or comprehensively evaluated for every possible downstream application.

Developers should take into account the known limitations of large language models, including performance variations across different languages, and carefully assess accuracy, safety, and fairness before deploying the model in sensitive or high-risk scenarios.
It is the responsibility of developers and downstream users to understand and comply with all applicable laws and regulations relevant to their use case, including but not limited to data protection, privacy, and content safety requirements.

Nothing in this Model Card should be interpreted as altering or restricting the terms of the MIT License under which the model is released.


## Citation

We kindly encourage citation of our work if you find it useful.

```
@misc{meituanlongcatteam2026longcatnextlexicalizingmodalitiesdiscrete,
      title={LongCat-Next: Lexicalizing Modalities as Discrete Tokens}, 
      author={Meituan LongCat Team},
      year={2026},
      eprint={2603.27538},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.27538}, 
}
```


## Contact
Please contact us at <a href="mailto:longcat-team@meituan.com">longcat-team@meituan.com</a> or open an issue if you have any questions.

#### WeChat Group
<img src="https://raw.githubusercontent.com/meituan-longcat/LongCat-Flash-Chat/main/wechat-assets/Wechat.png" width="200px">
