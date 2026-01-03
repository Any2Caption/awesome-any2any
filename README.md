<div align="center">
<h1> Awesome-Any-to-Any-Generation </h1> 
</div>


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



# 🌷 Datasets 

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

# Papers


## Any-to-Any

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


- [**4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities**](https://arxiv.org/pdf/2406.09406) [![Paper](https://img.shields.io/badge/NIPS24-CD5C5C2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://4m.epfl.ch/)   
    *🏷️:* `masked modeling`|`transformer encoder-decoder`|`📄🎨`

- [**C3LLM: Conditional Multimodal Content Generation Using Large Language Models**](https://arxiv.org/pdf/2405.16136)  [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()  
    *🏷️:* `transformer encoder-decoder`|`📄🎬🔊` 

- [**M2UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models**](https://arxiv.org/pdf/2311.11255)  [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()  
    *🏷️:* `llm`|`📄🎬🎨🔊🎶` 

- [**MuMu-LLaMA: Multi-modal Music Understanding and Generation via Large Language Models**](https://arxiv.org/pdf/2412.06660)  [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()   [![Star](https://img.shields.io/github/stars/shansongliu/MuMu-LLaMA.svg?style=social&label=Star)](https://github.com/shansongliu/MuMu-LLaMA)   
    *🏷️:* `llm`|`📄🎬🎨🔊🎶` 

- [**Visual Echoes: A Simple Unified Transformer for Audio-Visual Generation**](https://arxiv.org/pdf/2405.14598)  [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()  
    *🏷️:* `transformer encoder-decoder`|`📄🎨🔊` 


- [**CoDi: Any-to-Any Generation via Composable Diffusion**](https://arxiv.org/abs/2305.11846)   [![Paper](https://img.shields.io/badge/NIPS23-CD5C5C2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://codi-gen.github.io/)  [![Star](https://img.shields.io/github/stars/microsoft/i-Code.svg?style=social&label=Star)](https://github.com/microsoft/i-Code/tree/main/i-Code-V3)   
    *🏷️:* `diffusion`|`📄🎬🎨🔊`


- [**4M: Massively Multimodal Masked Modeling**](https://arxiv.org/pdf/2312.06647) [![Paper](https://img.shields.io/badge/NIPS23-CD5C5C2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://4m.epfl.ch/)   
    *🏷️:* `masked modeling`|`transformer encoder-decoder`|`📄🎨`

## Any-to-X (output-centric)


### Any-to-Text


- [**InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency**](https://arxiv.org/pdf/2508.18265) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()  [![Star](https://img.shields.io/github/stars/OpenGVLab/InternVL.svg?style=social&label=Star)](https://github.com/OpenGVLab/InternVL)    
    *🏷️:* `llm`|`📄🎬🎨`

- [**InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks**](https://arxiv.org/pdf/2312.14238)  [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]()  [![Star](https://img.shields.io/github/stars/OpenGVLab/InternVL.svg?style=social&label=Star)](https://github.com/OpenGVLab/InternVL)    
    *🏷️:* `llm`|`📄🎬🎨`

- [**X-InstructBLIP: A Framework for Aligning Image, 3D, Audio, Video to LLMs and its Emergent Cross-modal Reasoning**](https://arxiv.org/pdf/2311.18799) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://artemisp.github.io/X-InstructBLIP-page/) [![Star](https://img.shields.io/github/stars/salesforce/LAVIS.svg?style=social&label=Star)](https://github.com/salesforce/LAVIS/tree/main/projects/xinstructblip)   
    *🏷️:* `llm`|`📄🎬🎨🔊🧊`  

- [**Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**](https://arxiv.org/pdf/2306.02858) [![Paper](https://img.shields.io/badge/EMNLP23-b22222)]()   [![Star](https://img.shields.io/github/stars/DAMO-NLP-SG/Video-LLaMA.svg?style=social&label=Star)](https://github.com/DAMO-NLP-SG/Video-LLaMA)   
    *🏷️:* `llm`|`📄🎬🎨🔊`

- [**BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs**](https://arxiv.org/pdf/2307.08581)  [![Paper](https://img.shields.io/badge/arXiv23-b22222)]()  [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://bubo-gpt.github.io/) [![Star](https://img.shields.io/github/stars/magic-research/bubogpt.svg?style=social&label=Star)](https://github.com/magic-research/bubogpt)   
    *🏷️:* `llm`|`📄🎨🔊`

- [**AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model**](https://arxiv.org/pdf/2309.16058)  [![Paper](https://img.shields.io/badge/arXiv23-b22222)]()   
    *🏷️:* `llm`|`📄🎬🎨🔊`


- [**X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages**](https://arxiv.org/pdf/2305.04160) [![Paper](https://img.shields.io/badge/arXiv23-b22222)]()   [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://x-llm.github.io/)   [![Star](https://img.shields.io/github/stars/phellonchen/X-LLM.svg?style=social&label=Star)](https://github.com/phellonchen/X-LLM)   
    *🏷️:* `llm`|`📄🎬🎨🔊`

### Any-to-Image

- [**Any2AnyTryon: Leveraging Adaptive Position Embeddings for Versatile Virtual Clothing Tasks**](https://arxiv.org/pdf/2501.15891)  [![Paper](https://img.shields.io/badge/ICCV24-00CED1)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://logn-2024.github.io/Any2anyTryon/)  [![Star](https://img.shields.io/github/stars/logn-2024/Any2anyTryon.svg?style=social&label=Star)](https://github.com/logn-2024/Any2anyTryon) 


### Any-to-Video

- [**Videopoet:A large language model for zero-shot video generation**](https://arxiv.org/pdf/2312.14125)  [![Paper](https://img.shields.io/badge/ICCV24-00CED1)]() [![Star](https://img.shields.io/github/stars/Alpha-VLLM/Lumina-T2X.svg?style=social&label=Star)](https://github.com/Alpha-VLLM/Lumina-T2X)  


## X-to-Any (input-centric)
### Text-to-Any

- [**Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers**]() [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()   
    *🏷️:* `Diffusion`|`🎬🎨🔊🎶🎤`

### Image-to-Any

---

# 🐱‍🚀 Miscellaneous

## Workshop

## Survey

- [**MM-LLMs: Recent Advances in MultiModal Large Language Models**](https://arxiv.org/pdf/2401.13601)

- [**Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy**](https://arxiv.org/pdf/2401.00430)

- 

## Insteresting Works

- [**Awesome-Any-to-Any-Generation**](https://github.com/macabdul9/Awesome-Any-to-Any-Generation)
- [**Awesome-Multimodal-Large-Language-Models**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
- [**Awesome-Unified-Multimodal-Models**](https://github.com/showlab/Awesome-Unified-Multimodal-Models)
- [**Awesome-Unified-Multimodal-Models**](https://github.com/AIDC-AI/Awesome-Unified-Multimodal-Models)
- [**Awesome-Anything**](https://github.com/VainF/Awesome-Anything)
    <details><summary>general AI methods for Anything</summary>A curated list of general AI methods for Anything: AnyObject, AnyGeneration, AnyModel, AnyTask, etc.</details>

# ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Any2Caption/awesome-any2any&type=date&legend=top-left)](https://www.star-history.com/#Any2Caption/awesome-any2any&type=date&legend=top-left)
