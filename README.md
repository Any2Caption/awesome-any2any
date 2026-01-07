<div align="center">
<h1> Awesome-Any-to-Any-Generation </h1> 
</div>


# 📣News
✨✨✨We are organizing a **CVPR26 Workshop** on [Any-to-Any Multimodal Learning](https://a2a-mml-2026.vercel.app/), welcome submission.


# 🎨 Introduction
Traditional generative models are typically designed for a fixed input–output modality pair (e.g., text-to-image or image-to-text).
However, real-world multimodal intelligence requires the ability to flexibly generate across arbitrary modality combinations, including multi-input and multi-output settings.

This repository aims to systematize **`Any-to-Any Generation`**, where models can accept inputs from arbitrary modalities and produce outputs in arbitrary modalities within a unified framework.

  <p align="center">
  <img src="assets/introduction.png" width="100%">
</p>

### What qualifies as Any-to-Any Generation?

A model/system is considered Any-to-Any if it satisfies at least one of the following:

1. Supports arbitrary combinations of input modalities and output modalities within a single unified framework;
2. Enables multi-input and/or multi-output generation without task-specific retraining;
3. Relies on a modality-agnostic intermediate representation (e.g., shared latent space, discrete tokens, structured programs);
4. Demonstrates compositional generalization to unseen modality mappings.


# 📕 Table of Content

- [🌷 Datasets](#-dataset)
- [📃Papers](#-papers)
    - [Any-to-Any](#any-to-any)
    - [Any-to-X (output-centric)](#any-to-x-output-centric)
        - [Any-to-Text](#any-to-text)
        - [Any-to-Image](#any-to-image)
        - [Any-to-Video](#any-to-video)
    - [X-to-Any (input-centric)](#x-to-any-input-centric)
        - [Text-to-Any](#text-to-any)
        - [Image-to-Any](#image-to-any)
- [🐱‍🚀 Miscellaneous](#)
    - [Workshop](#workshop)
    - [Survey](#survey)
    - [Interesting Works](#insteresting-works)



# 🌷 Datasets 

- [**MMIE: Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models**](https://arxiv.org/pdf/2410.10139())  `📄🎨`

- [**Interleaved Scene Graphs for Interleaved Text-and-Image Generation Assessment**](https://openreview.net/pdf?id=rDLgnYLM5b) `📄🎨`

- [**OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation**](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_OpenING_A_Comprehensive_Benchmark_for_Judging_Open-ended_Interleaved_Image-Text_Generation_CVPR_2025_paper.pdf)  `📄🎨`

- [**A High-Quality Dataset and Reliable Evaluation for Interleaved Image-Text Generation**](https://arxiv.org/pdf/2506.09427)   `📄🎨`

- [**InterleavedBench: Holistic Evaluation for Interleaved Text-and-Image Generation**](https://arxiv.org/pdf/2406.14643)  `📄🎨`

- [**OmniBench: Towards The Future of Universal Omni-Language Models**](https://arxiv.org/pdf/2409.15272)    `📄🎨🔊🎶`

- [**Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities**](https://arxiv.org/pdf/2505.17862)     `📄🎬🔊`

- [**AV-Reasoner: Improving and Benchmarking Clue-Grounded Audio-Visual Counting for MLLMs**](https://arxiv.org/pdf/2506.05328)      `📄🎬🔊`


<!-- CVPR-8A2BE2 -->
<!-- WACV-6a5acd -->
<!-- NIPS-CD5C5C2 -->
<!-- ICML-FF7F50 -->
<!-- ICCV-00CED1 -->
<!-- ECCV-1e90ff -->
<!-- TPAMI-BC8F8F -->
<!-- IJCAI-228b22 -->
<!-- AAAI-c71585 -->
<!-- arXiv-b22222 -->
<!-- ACL-191970 -->
<!-- TPAMI-ffa07a -->

# 📃Papers


## Any-to-Any
Any-to-Any generation refers to unified systems that can take inputs from multiple modalities (e.g., text/image/video/audio) and produce outputs in multiple modalities within a single framework.

- [**Symbolic Representation for Any-to-Any Generative Tasks**](https://arxiv.org/pdf/2504.17261)  [![Paper](https://img.shields.io/badge/CVPR25-8A2BE2)]()  [![Star](https://img.shields.io/github/stars/Jiaqi-Chen-00/Any-2-Any.svg?style=social&label=Star)](https://github.com/Jiaqi-Chen-00/Any-2-Any)    
    *🏷️:* `llm`|`diffusion`|`📄🎬🎨🧊`

- [**Easy, fast, and cheap omni-modality model serving for everyone**](https://docs.vllm.ai/projects/vllm-omni/en/latest/#about) [![Star](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social&label=Star)](https://github.com/vllm-project/vllm)   
    *🏷️:* `mllm`|`Talker`|`📄🎬🎨🔊`

- [**Uni-MoE-2.0-Omni: Scaling Language-Centric Omnimodal Large Model with Advanced MoE, Training and Data**](https://arxiv.org/pdf/2511.12609) ![Paper](https://img.shields.io/badge/arXiv25-b22222)  [![Star](https://img.shields.io/github/stars/HITsz-TMG/Uni-MoE.svg?style=social&label=Star)](https://github.com/HITsz-TMG/Uni-MoE/tree/master/Uni-MoE-2)   
    🏷️:* `llm`|`moe`|`📄🎬🎨🔊🎤`

- [**OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM**](https://arxiv.org/pdf/2510.15870) ![Paper](https://img.shields.io/badge/arXiv25-b22222) [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://nvlabs.github.io/OmniVinci/)    [![Star](https://img.shields.io/github/stars/NVlabs/OmniVinci.svg?style=social&label=Star)](https://github.com/NVlabs/OmniVinci)  
    🏷️:* `llm`|`MMDiT`|`📄🎬🎨🔊🎤`

- [**Ming-Omni: A Unified Multimodal Model for Perception and Generation**](https://arxiv.org/pdf/2506.09344)  ![Paper](https://img.shields.io/badge/arXiv25-b22222) [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://lucaria-academy.github.io/Ming-Omni/)    [![Star](https://img.shields.io/github/stars/inclusionAI/Ming.svg?style=social&label=Star)](https://github.com/inclusionAI/Ming)  
    🏷️:* `Ling`|`flow`|`📄🎬🎨🔊🎤`

- [**Qwen2.5-Omni Technical Report**](https://arxiv.org/pdf/2503.20215) ![Paper](https://img.shields.io/badge/arXiv25-b22222)  [![Star](https://img.shields.io/github/stars/QwenLM/Qwen2.5-Omni.svg?style=social&label=Star)](https://github.com/QwenLM/Qwen2.5-Omni)   
    🏷️:* `llm`|`flow`|`📄🎬🎨🔊🎶🎤`

- [**Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities**](https://arxiv.org/pdf/2410.11190) ![Paper](https://img.shields.io/badge/arXiv25-b22222)  [![Star](https://img.shields.io/github/stars/gpt-omni/mini-omni2.svg?style=social&label=Star)](https://github.com/gpt-omni/mini-omni2)   
    🏷️:* `mllm`|`📄🎬🎨🎶🎤`

- [**MiniCPM-o 2.6: A GPT-4o Level MLLM for Vision, Speech, and Multimodal Live Streaming on Your Phone**](https://openbmb.vercel.app/minicpm-o-2-6-en)     
    🏷️:* `mllm`|`📄🎬🎨🎤`

- [**Baichuan-Omni-1.5 Technical Report**](https://arxiv.org/pdf/2501.15368) ![Paper](https://img.shields.io/badge/arXiv25-b22222)  [![Star](https://img.shields.io/github/stars/baichuan-inc/Baichuan-Omni-1.5.svg?style=social&label=Star)](https://github.com/baichuan-inc/Baichuan-Omni-1.5)   
    🏷️:* `mllm`|`📄🎬🎨🎤`

- [**Show-o2: Improved Native Unified Multimodal Models**](https://arxiv.org/pdf/2506.15564)  ![Paper](https://img.shields.io/badge/arXiv25-b22222)  [![Star](https://img.shields.io/github/stars/showlab/Show-o.svg?style=social&label=Star)](https://github.com/showlab/Show-o)   
    *🏷️:* `llm`|`flow`|`📄🎬🎨`

- [**Show-o: One Single Transformer to Unify Multimodal Understanding and Generation**](https://arxiv.org/pdf/2408.12528)  ![Paper](https://img.shields.io/badge/ICLR25-696969)  [![Star](https://img.shields.io/github/stars/showlab/Show-o.svg?style=social&label=Star)](https://github.com/showlab/Show-o)   
    *🏷️:* `llm`|`diffusion`|`📄🎬🎨`

- [**CoDi2: In-Context, Interleaved, and Interactive Any-to-Any Generation**](https://arxiv.org/abs/2311.18775)  [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://codi-2.github.io/)  [![Star](https://img.shields.io/github/stars/microsoft/i-Code.svg?style=social&label=Star)](https://github.com/microsoft/i-Code/tree/main/CoDi-2)    
    *🏷️:* `llm`|`diffusion`|`📄🎬🎨🔊`


- [**Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action**](https://arxiv.org/pdf/2312.17172)  [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://unified-io-2.allenai.org/)  [![Star](https://img.shields.io/github/stars/allenai/unified-io-2.svg?style=social&label=Star)](https://github.com/allenai/unified-io-2)  
    *🏷️:* `transformer encoder-decoder`|`📄🎬🎨🔊🤖`


- [**C3Net: Compound Conditioned ControlNet for Multimodal Content Generation**](https://arxiv.org/pdf/2311.17951)  [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://unified-io-2.allenai.org/)  [![Star](https://img.shields.io/github/stars/JordanZh/C3Net.svg?style=social&label=Star)](https://github.com/JordanZh/C3Net)  
    *🏷️:* `diffusion`|`📄🎨🔊`


- [**Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners**](https://arxiv.org/pdf/2402.17723)  [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://yzxing87.github.io/Seeing-and-Hearing/)  [![Star](https://img.shields.io/github/stars/yzxing87/Seeing-and-Hearing.svg?style=social&label=Star)](https://github.com/yzxing87/Seeing-and-Hearing)  
    *🏷️:* `diffusion`|`🎬🎨🔊`


- [**ModaVerse: Efficiently Transforming Modalities with LLMs**](https://arxiv.org/pdf/2401.06395)  [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://xinke-wang.github.io/modaverse)  [![Star](https://img.shields.io/github/stars/xinke-wang/ModaVerse.svg?style=social&label=Star)](https://github.com/xinke-wang/ModaVerse)  
    *🏷️:* `llm`|`diffusion`|`📄🎬🎨🔊`


- [**NExT-GPT: Any-to-Any Multimodal LLM**](https://arxiv.org/pdf/2309.05519)  [![Paper](https://img.shields.io/badge/ICML24-FF7F50)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://next-gpt.github.io/)  [![Star](https://img.shields.io/github/stars/NExT-GPT/NExT-GPT.svg?style=social&label=Star)](https://github.com/NExT-GPT/NExT-GPT)    
    *🏷️:* `llm`|`diffusion`|`📄🎬🎨🔊`


- [**AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling**](https://arxiv.org/abs/2402.12226) [![Paper](https://img.shields.io/badge/ACL24-191970)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://junzhan2000.github.io/AnyGPT.github.io/)  [![Star](https://img.shields.io/github/stars/OpenMOSS/AnyGPT.svg?style=social&label=Star)](https://github.com/OpenMOSS/AnyGPT)  
    *🏷️:* `llm`|`tokenizer`|`📄🎨🎶🎤`


- [**Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks**](https://arxiv.org/pdf/2206.08916)    [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://unified-io.allenai.org/)  [![Star](https://img.shields.io/github/stars/allenai/unified-io-inference.svg?style=social&label=Star)](https://github.com/allenai/unified-io-inference)   
    *🏷️:* `transformer encoder-decoder`|`📄🎨`


- [**4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities**](https://arxiv.org/pdf/2409.01392) [![Paper](https://img.shields.io/badge/NIPS24-CD5C5C2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://4m.epfl.ch/)   
    *🏷️:* `masked modeling`|`transformer encoder-decoder`|`📄🎨`

- [**ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems**](https://arxiv.org/pdf/2405.16136)  [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()  
    *🏷️:* `agent`|`📄🎬🎨` 


- [**X-VILA: Cross-Modality Alignment for Large Language Model**](https://arxiv.org/pdf/2405.19335)  [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()  
    *🏷️:* `llm`|`diffusion`|`📄🎬🎨🔊` 

- [**C3LLM: Conditional Multimodal Content Generation Using Large Language Models**](https://arxiv.org/pdf/2405.16136)  [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()  
    *🏷️:* `transformer encoder-decoder`|`📄🎬🔊` 

- [**M2UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models**](https://arxiv.org/pdf/2311.11255)  [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()  
    *🏷️:* `llm`|`📄🎬🎨🔊🎶` 

- [**TEAL: Tokenize and Embed ALL for Multi-modal Large Language Models**](https://arxiv.org/pdf/2311.04589)  [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()  
    *🏷️:* `mllm`|`📄🎬🎨🔊🎤` 

- [**MuMu-LLaMA: Multi-modal Music Understanding and Generation via Large Language Models**](https://arxiv.org/pdf/2412.06660)  [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()   [![Star](https://img.shields.io/github/stars/shansongliu/MuMu-LLaMA.svg?style=social&label=Star)](https://github.com/shansongliu/MuMu-LLaMA)   
    *🏷️:* `mllm`|`📄🎬🎨🔊🎶` 

- [**Visual Echoes: A Simple Unified Transformer for Audio-Visual Generation**](https://arxiv.org/pdf/2405.14598)  [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()  
    *🏷️:* `transformer encoder-decoder`|`📄🎨🔊` 

- [**AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head**](https://arxiv.org/pdf/2304.12995)  [![Paper](https://img.shields.io/badge/AAAI24-191970)]() [![Star](https://img.shields.io/github/stars/AIGC-Audio/AudioGPT.svg?style=social&label=Star)](https://github.com/AIGC-Audio/AudioGPT)     
    *🏷️:* `llm`|`📄🎬🔊🎤🎶` 

- [**HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**](https://arxiv.org/pdf/2303.17580)  [![Paper](https://img.shields.io/badge/NIPS23-CD5C5C2)]()   [![Star](https://img.shields.io/github/stars/microsoft/JARVIS.svg?style=social&label=Star)](https://github.com/microsoft/JARVIS)  
    *🏷️:* `llm`|`📄🎬🎨🔊🎤` 

- [**CoDi: Any-to-Any Generation via Composable Diffusion**](https://arxiv.org/abs/2305.11846)   [![Paper](https://img.shields.io/badge/NIPS23-CD5C5C2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://codi-gen.github.io/)  [![Star](https://img.shields.io/github/stars/microsoft/i-Code.svg?style=social&label=Star)](https://github.com/microsoft/i-Code/tree/main/i-Code-V3)   
    *🏷️:* `diffusion`|`📄🎬🎨🔊`


- [**4M: Massively Multimodal Masked Modeling**](https://arxiv.org/pdf/2312.06647) [![Paper](https://img.shields.io/badge/NIPS23-CD5C5C2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://4m.epfl.ch/)   
    *🏷️:* `masked modeling`|`transformer encoder-decoder`|`📄🎨`


## Any-to-X (output-centric)
Any-to-X methods accept flexible inputs (potentially multi-modal, such as text + image + audio) but generate a single target modality. This setting is often practically useful (e.g., “any condition → text report”, “any condition → image synthesis”, “any condition → video generation”), and it highlights how systems fuse heterogeneous conditions and maintain faithfulness to each input. Compared to fully general Any-to-Any systems, Any-to-X typically has a simpler decoding interface, but still demands strong cross-modal alignment and robust conditioning mechanisms.

### Any-to-Text
Any-to-Text focuses on producing textual outputs (captioning, explanation, dialogue, reasoning traces, instruction-following) from arbitrary visual/audio/3D/video inputs.


- [**Omni-R1: Reinforcement Learning for Omnimodal Reasoning via Two-System Collaboration**](https://arxiv.org/pdf/2505.20256)   [![Paper](https://img.shields.io/badge/NIPS25-CD5C5C2)]()  [![Star](https://img.shields.io/github/stars/aim-uofa/Omni-R1.svg?style=social&label=Star)](https://github.com/aim-uofa/Omni-R1)         
    *🏷️:* `Qwen2.5-Omni-7B`|`📄🎬🎨🔊🎤`

- [**Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs**]() [![Paper](https://img.shields.io/badge/arXiv25-b22222)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)    
    *🏷️:* `mllm`|`📄🎨🎤` 

- [**A Reason-then-Describe Instruction Interpreter for Controllable Video Generation**](https://arxiv.org/pdf/2511.20563)  [![Paper](https://img.shields.io/badge/arXiv25-b22222)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://sqwu.top/ReaDe/)     
    *🏷️:* `Qwen2.5-Omni-7B`|`📄🎬🎨🎥🏃🏻🔊🎤`

- [**Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities**](https://arxiv.org/pdf/2505.17862)  [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()  [![Star](https://img.shields.io/github/stars/Lliar-liar/Daily-Omni.svg?style=social&label=Star)](https://github.com/Lliar-liar/Daily-Omni)         
    *🏷️:* `Qwen2.5-VL/Qwen2-Audio`|`📄🎬🎨🔊🎤`

- [**Any2Caption:Interpreting Any Condition to Caption for Controllable Video Generation**](https://arxiv.org/pdf/2503.24379)  [![Paper](https://img.shields.io/badge/arXiv25-b22222)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://sqwu.top/Any2Cap/)     
    *🏷️:* `Qwen2.5-VL`|`📄🎬🎨🎥🏃🏻`

- [**InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency**](https://arxiv.org/pdf/2508.18265) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()  [![Star](https://img.shields.io/github/stars/OpenGVLab/InternVL.svg?style=social&label=Star)](https://github.com/OpenGVLab/InternVL)    
    *🏷️:* `mllm`|`📄🎬🎨`

- [**InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks**](https://arxiv.org/pdf/2312.14238)  [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]()  [![Star](https://img.shields.io/github/stars/OpenGVLab/InternVL.svg?style=social&label=Star)](https://github.com/OpenGVLab/InternVL)    
    *🏷️:* `mllm`|`📄🎬🎨`

- [**EMU: GENERATIVE PRETRAINING IN MULTIMODALITY**](https://arxiv.org/pdf/2312.14238)  [![Paper](https://img.shields.io/badge/ICLR24-696969)]()  [![Star](https://img.shields.io/github/stars/baaivision/Emu.svg?style=social&label=Star)](https://github.com/baaivision/Emu)    
    *🏷️:* `mllm`|`📄🎬🎨`

- [**Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts**](https://arxiv.org/pdf/2405.11273) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]() [![Star](https://img.shields.io/github/stars/HITsz-TMG/Uni-MoE.svg?style=social&label=Star)](https://github.com/HITsz-TMG/Uni-MoE)   
    *🏷️:* `llm`|`moe`|`📄🎬🎨🔊🎤`  

- [**X-InstructBLIP: A Framework for Aligning Image, 3D, Audio, Video to LLMs and its Emergent Cross-modal Reasoning**](https://arxiv.org/pdf/2311.18799) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://artemisp.github.io/X-InstructBLIP-page/) [![Star](https://img.shields.io/github/stars/salesforce/LAVIS.svg?style=social&label=Star)](https://github.com/salesforce/LAVIS/tree/main/projects/xinstructblip)   
    *🏷️:* `llm`|`📄🎬🎨🔊🧊`  

- [**What Matters in Training a GPT4-Style Language Model with Multimodal Inputs?**](https://arxiv.org/pdf/2307.02469) [![Paper](https://img.shields.io/badge/NAACL24-191970)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://lynx-llm.github.io/)  [![Star](https://img.shields.io/github/stars/bytedance/lynx-llm.svg?style=social&label=Star)](https://github.com/bytedance/lynx-llm)   
    *🏷️:* `llm`|`📄🎬🎨`

- [**Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**](https://arxiv.org/pdf/2306.02858) [![Paper](https://img.shields.io/badge/EMNLP23-191970)]()   [![Star](https://img.shields.io/github/stars/DAMO-NLP-SG/Video-LLaMA.svg?style=social&label=Star)](https://github.com/DAMO-NLP-SG/Video-LLaMA)   
    *🏷️:* `llm`|`📄🎬🎨🔊`

- [**BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs**](https://arxiv.org/pdf/2307.08581)  [![Paper](https://img.shields.io/badge/arXiv23-b22222)]()  [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://bubo-gpt.github.io/) [![Star](https://img.shields.io/github/stars/magic-research/bubogpt.svg?style=social&label=Star)](https://github.com/magic-research/bubogpt)   
    *🏷️:* `llm`|`📄🎨🔊`

- [**AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model**](https://arxiv.org/pdf/2309.16058)  [![Paper](https://img.shields.io/badge/arXiv23-b22222)]()   
    *🏷️:* `llm`|`📄🎬🎨🔊`


- [**X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages**](https://arxiv.org/pdf/2305.04160) [![Paper](https://img.shields.io/badge/arXiv23-b22222)]()   [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://x-llm.github.io/)   [![Star](https://img.shields.io/github/stars/phellonchen/X-LLM.svg?style=social&label=Star)](https://github.com/phellonchen/X-LLM)   
    *🏷️:* `llm`|`📄🎬🎨🔊`

- [**ONE-PEACE: Exploring One General Representation Model Toward Unlimited Modalities**](https://arxiv.org/pdf/2305.11172) [![Paper](https://img.shields.io/badge/arXiv23-b22222)]()  [![Star](https://img.shields.io/github/stars/OFA-Sys/ONE-PEACE.svg?style=social&label=Star)](https://github.com/OFA-Sys/ONE-PEACE)   
    *🏷️:* `llm`|`modality alignment`|`📄🎬🎨🔊`

### Any-to-Image
Any-to-Image methods generate images conditioned on diverse inputs beyond text, such as images, sketches, poses, layouts, audio cues, or multi-modal prompts.

- [**OmniGen-AR: AutoRegressive Any-to-Image Generation**](https://openreview.net/pdf?id=Gxw10T7uOm)  [![Paper](https://img.shields.io/badge/NIPS25-CD5C5C2)]()

- [**Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model**](https://arxiv.org/pdf/2404.09967) [![Paper](https://img.shields.io/badge/ICLR25-696969)]() [![Star](https://img.shields.io/github/stars/HL-hanlin/Ctrl-Adapter.svg?style=social&label=Star)](https://github.com/HL-hanlin/Ctrl-Adapter) 

- [**OmniGen: Unified Image Generation**](https://arxiv.org/pdf/2409.11340v2)  [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]()  

- [**Any2AnyTryon: Leveraging Adaptive Position Embeddings for Versatile Virtual Clothing Tasks**](https://arxiv.org/pdf/2501.15891)  [![Paper](https://img.shields.io/badge/ICCV24-2f4f4f)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://logn-2024.github.io/Any2anyTryon/)  [![Star](https://img.shields.io/github/stars/logn-2024/Any2anyTryon.svg?style=social&label=Star)](https://github.com/logn-2024/Any2anyTryon) 


### Any-to-Video
Any-to-Video targets video generation from flexible conditions (text/image/video/audio/trajectory/layout).

- [**Videopoet:A large language model for zero-shot video generation**](https://arxiv.org/pdf/2312.14125)  [![Paper](https://img.shields.io/badge/ICCV24-2f4f4f)]() [![Star](https://img.shields.io/github/stars/Alpha-VLLM/Lumina-T2X.svg?style=social&label=Star)](https://github.com/Alpha-VLLM/Lumina-T2X)  

- [**VideoComposer: Compositional Video Synthesis with Motion Controllability**](https://arxiv.org/pdf/2306.02018.pdf) [![Paper](https://img.shields.io/badge/arXiv23-b22222)]() [![Star](https://img.shields.io/github/stars/ali-vilab/videocomposer.svg?style=social&label=Star)](https://github.com/ali-vilab/videocomposer)  


## X-to-Any (input-centric)
X-to-Any methods start from a fixed input modality but aim to generate multiple output modalities (e.g., text → image/video/audio; image → text/video/audio). This setting is useful for studying whether a model learns a shared multimodal representation that can be decoded into different modalities. Compared to Any-to-X, the emphasis is on multi-head decoding and output diversity, often requiring modality-specific decoders while sharing a common backbone or latent space.

### Text-to-Any
Text-to-Any expands classic text-to-image into text-conditioned generation across multiple modalities, such as video, audio, music, speech, and even structured outputs. Typical solutions include unified diffusion/flow backbones, discrete token modeling, or LLM-centered generation that routes to modality experts.

- [**Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers**]() [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()   
    *🏷️:* `Diffusion`|`🎬🎨🔊🎶🎤`

### Image-to-Any
Image-to-Any aims to generate other modalities from visual input, such as image → text (captioning/VQA), image → video (animation), image → audio (foley/sound), or image → 3D (reconstruction). The main technical challenge is learning mappings from static visual cues to modalities with missing dimensions (e.g., time, sound source, geometry), which often requires strong priors, world knowledge, or intermediate structured representations.


## Any Alignment

- [**LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment**](https://arxiv.org/pdf/2310.01852) [![Paper](https://img.shields.io/badge/ICLR24-8A2BE2)]()   [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/LanguageBind.svg?style=social&label=Star)](https://github.com/PKU-YuanGroup/LanguageBind)   
    *🏷️:* ` Binding modalities with languages`|`📄🎨🎬🔊`

- [**ImageBind: One Embedding Space To Bind Them All**](https://arxiv.org/pdf/2305.05665) [![Paper](https://img.shields.io/badge/CVPR23-8A2BE2)]()   [![Star](https://img.shields.io/github/stars/facebookresearch/ImageBind.svg?style=social&label=Star)](https://github.com/facebookresearch/ImageBind)   
    *🏷️:* ` Binding modalities with images`|`📄🎨🎬🔊`

---

# 🐱‍🚀 Miscellaneous

## Workshop

- [Any-to-Any Multimodal Learning CVPR Workshop 2026](https://a2a-mml-2026.vercel.app/)


## Survey

- [**Unified multimodal understanding and generation models: Advances, challenges, and opportunities**](https://arxiv.org/pdf/2505.02567)

- [**A Survey of Unified Multimodal Understanding and Generation: Advances and Challenges**](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176289261.16802577)

- [**On path to multimodal generalist: General-level and general-bench**](https://openreview.net/pdf?id=VsJ1K2HV3k)

- [**MM-LLMs: Recent Advances in MultiModal Large Language Models**](https://arxiv.org/pdf/2401.13601)

- [**Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy**](https://arxiv.org/pdf/2401.00430)

- [**Multimodal Foundation Models: From Specialists to General-Purpose Assistants**](https://arxiv.org/pdf/2309.10020)



## Awesome Github Repo

- [**Awesome-Any-to-Any-Generation**](https://github.com/macabdul9/Awesome-Any-to-Any-Generation)

- [**Awesome-Multimodal-Large-Language-Models**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)

- [**Awesome-Unified-Multimodal-Models**](https://github.com/showlab/Awesome-Unified-Multimodal-Models)

- [**LLMs Meet Multimodal Generation and Editing: A Survey**](https://github.com/YingqingHe/Awesome-LLMs-meet-Multimodal-Generation)

- [**Awesome-Unified-Multimodal-Models**](https://github.com/AIDC-AI/Awesome-Unified-Multimodal-Models)

- [**Awesome Autoregressive Models in Vision**](https://github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey)

- [**Awesome-Anything**](https://github.com/VainF/Awesome-Anything)
    <details><summary>general AI methods for Anything</summary>A curated list of general AI methods for Anything: AnyObject, AnyGeneration, AnyModel, AnyTask, etc.</details>


## Interesting Works
- [**Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models**](https://arxiv.org/pdf/2510.08492)
- [**UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes**](https://arxiv.org/pdf/2205.10337)


# ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=any2any-mllm/awesome-any2any&type=date&legend=top-left)](https://www.star-history.com/#any2any-mllm/awesome-any2any&type=date&legend=top-left)
