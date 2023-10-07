<img src='title.png' align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2310.01830-b31b1b.svg)](https://arxiv.org/abs/2310.01830)
[![Survey](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) 
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) 
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)  
This project is associated with our survey paper which comprehensively contextualizes the advance of the recent **AI**-**G**enerated Images as Data **S**ource (**AIGS**) and visual AIGC by formulating taxonomies according to methodologies and applications.

<img src='teaser.png' align="center">

**AI-Generated Images as Data Source: The Dawn of Synthetic Era** [[Paper](https://arxiv.org/abs/2310.01830)]  
*Zuhao Yang, Fangneng Zhan, Kunhao Liu, Muyu Xu, Shijian Lu*  
arXiv, 2023 

<br>

[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) 
You are welcome to promote papers via pull request. <br>
The process to submit a pull request:
- a. Fork the project into your own repository.
- b. Add the Title, Author, Conference, Paper link, Project link, and Code link in `README.md` with below format:
```
**Title**<br>
*Author*<br>
Conference
[[Paper](Paper link)]
[[Project](Project link)]
[[Code](Code link)]
[[Video](Video link)]
```
- c. Submit the pull request to this branch.

<br>

## Related Surveys & Projects
**Machine Learning for Synthetic Data Generation: A Review**  
*Yingzhou Lu, Minjie Shen, Huazheng Wang, Wenqi Wei*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2302.04062)]

**Synthetic Data in Human Analysis: A Survey**  
*Indu Joshi, Marcel Grimmer, Christian Rathgeb, Christoph Busch, Francois Bremond, Antitza Dantcheva*  
arXiv 2022 [[Paper](https://arxiv.org/abs/2208.09191)]

**A Review of Synthetic Image Data and Its Use in Computer Vision**  
*Keith Man, Javaan Chahl*  
J. Imaging 2022 [[Paper](https://www.mdpi.com/2313-433X/8/11/310)]

**Survey on Synthetic Data Generation, Evaluation Methods and GANs**  
*Alvaro Figueira, Bruno Vaz*  
Mathematics 2022 [[Paper](https://www.mdpi.com/2227-7390/10/15/2733)]


## Table of Contents (Work in Progress)
Methods:
- [Generative Models](#GenerativeModels-link)
  - [Label Acquisition](#GenLabelAcquisition-link)
  - [Data Augmentation](#GenDataAugmentation-link)
- [Neural Rendering](#NeuralRendering-link)
  - [Label Acquisition](#NeuLabelAcquisition-link)
  - [Data Augmentation](#NeuDataAugmentation-link)

Applications:
- [2D Visual Perception](#2DVisualPerception-link)
  - [Image Classification](#Classification-link)
  - [Image Segmentation](#Segmentation-link)
  - [Object Detection](#Detection-link)
- [Visual Generation](#VisualGeneration-link)
- [Self-supervised Learning](#SelfsupervisedLearning-link)
- [3D Visual Perception](#3DVisualPerception-link)
  - [Robotics](#Robotics-link)
  - [Autonomous Driving](#AutonomousDriving-link)
- [Other Applications](#OtherApplications-link)
  - [Medical](#Medical-link)
  - [Testing Data](#Test-link)

Datasets:
- [Datasets](#Datasets-link)


# Methods

## Generative Models
<a id="GenerativeModels-link"></a>

### Label Acquisition
<a id="GenLabelAcquisition-link"></a>

**[BigGAN] Large Scale GAN Training for High Fidelity Natural Image Synthesis**  
*Andrew Brock, Jeff Donahue, Karen Simonyan*  
ICLR 2019 [[Paper](https://arxiv.org/abs/1809.11096)]

**[VQ-Diffusion] Vector Quantized Diffusion Model for Text-to-Image Synthesis**  
*Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, Baining Guo*  
CVPR 2022 [[Paper](https://arxiv.org/abs/2111.14822)][[Code](https://github.com/cientgu/VQ-Diffusion#vector-quantized-diffusion-model-for-text-to-image-synthesis-cvpr2022-oral)]

**[LDM] High-Resolution Image Synthesis with Latent Diffusion Models**  
*Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer*  
CVPR 2022 [[Paper](https://arxiv.org/abs/2112.10752)][[Code](https://github.com/CompVis/latent-diffusion)]

**[Imagen] Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding**  
*Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, Mohammad Norouzi*  
NeurIPS 2022 [[Paper](https://arxiv.org/abs/2205.11487)][[Project](https://imagen-ai.com/?v=0j)]

**[DALL-E 2] Hierarchical Text-Conditional Image Generation with CLIP Latents**  
*Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen*  
arXiv 2022 [[Paper](https://arxiv.org/abs/2204.06125)]

**GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**  
*Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, Mark Chen*  
ICML 2022 [[Paper](https://arxiv.org/abs/2112.10741)][[Code](https://github.com/openai/glide-text2im)]

**DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort**  
*Yuxuan Zhang, Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche, Adela Barriuso, Antonio Torralba, Sanja Fidler*   
CVPR 2021 [[Paper](https://arxiv.org/abs/2104.06490)][[Project](https://nv-tlabs.github.io/datasetGAN/)][[Code](https://github.com/nv-tlabs/datasetGAN_release/tree/master)]

**DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion Models**  
*Weijia Wu, Yuzhong Zhao, Hao Chen, Yuchao Gu, Rui Zhao, Yefei He, Hong Zhou, Mike Zheng Shou, Chunhua Shen*  
NeurIPS 2023 [[Paper](https://arxiv.org/abs/2308.06160)][[Project](https://weijiawu.github.io/DatasetDM_page/)][[Code](https://github.com/showlab/DatasetDM)]

**DALL-E for Detection: Language-driven Compositional Image Synthesis for Object Detection**  
*Yunhao Ge, Jiashu Xu, Brian Nlong Zhao, Neel Joshi, Laurent Itti, Vibhav Vineet*  
arXiv 2022 [[Paper](https://arxiv.org/abs/2206.09592)][[Code](https://github.com/gyhandy/Text2Image-for-Detection)]


### Data Augmentation
<a id="GenDataAugmentation-link"></a>

**[StyleGAN 2]Analyzing and Improving the Image Quality of StyleGAN**  
*Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila*  
CVPR 2020 [[Paper](https://arxiv.org/abs/1912.04958)][[Code](https://github.com/NVlabs/stylegan2)][[Video](https://www.youtube.com/watch?v=c-NJtV9Jvp0&feature=youtu.be)]

**GAN Augmentation: Augmenting Training Data using Generative Adversarial Networks**  
*Christopher Bowles, Liang Chen, Ricardo Guerrero, Paul Bentley, Roger Gunn, Alexander Hammers, David Alexander Dickie, Maria Valdés Hernández, Joanna Wardlaw, Daniel Rueckert*  
arXiv 2018 [[Paper](https://arxiv.org/abs/1810.10863)]

**Enhancement of Image Classification Using Transfer Learning and GAN-Based Synthetic Data Augmentation**  
*Subhajit Chatterjee, Debapriya Hazra, Yung-Cheol Byun, Yong-Woon Kim*  
Mathmatics 2022 [[Paper](https://www.mdpi.com/2227-7390/10/9/1541)]

**A data augmentation perspective on diffusion models and retrieval**  
*Max F. Burg, Florian Wenzel, Dominik Zietlow, Max Horn, Osama Makansi, Francesco Locatello, Chris Russell*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2304.10253)]

**Effective Data Augmentation With Diffusion Models**  
*Brandon Trabucco, Kyle Doherty, Max Gurinas, Ruslan Salakhutdinov*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2302.07944)][[Project](http://btrabuc.co/da-fusion/)]

**Diversify Your Vision Datasets with Automatic Diffusion-Based Augmentation**  
*Lisa Dunlap, Alyssa Umino, Han Zhang, Jiezhi Yang, Joseph E. Gonzalez, Trevor Darrell*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2305.16289)][[Code](https://github.com/lisadunlap/ALIA)]


## Neural Rendering
<a id="NeuralRendering-link"></a>

### Label Acquisition
<a id="NeuLabelAcquisition-link"></a>

**VMRF: View Matching Neural Radiance Fields**  
*Jiahui Zhang, Fangneng Zhan, Rongliang Wu, Yingchen Yu, Wenqing Zhang, Bai Song, Xiaoqin Zhang, Shijian Lu*  
ACM MM 2022 [[Paper](https://arxiv.org/abs/2207.02621)]

**NeRF-Supervision: Learning Dense Object Descriptors from Neural Radiance Fields**  
*Thomas Lips, Victor-Louis De Gusseme, Francis wyffels*  
ICRA 2022 [[Paper](https://arxiv.org/abs/2203.01913)][[Project](https://yenchenlin.me/nerf-supervision/)][[Code](https://github.com/yenchenlin/nerf-supervision-public)][[Video](https://www.youtube.com/watch?v=_zN-wVwPH1s)]


### Data Augmentation
<a id="NeuDataAugmentation-link"></a>

**Neural-Sim: Learning to Generate Training Data with NeRF**  
*Yunhao Ge, Harkirat Behl, Jiashu Xu, Suriya Gunasekar, Neel Joshi, Yale Song, Xin Wang, Laurent Itti, Vibhav Vineet*  
ECCV 2022 [[Paper](https://arxiv.org/abs/2207.11368)][[Code](https://github.com/gyhandy/Neural-Sim-NeRF)]

**3D Data Augmentation for Driving Scenes on Camera**  
*Wenwen Tong, Jiangwei Xie, Tianyu Li, Hanming Deng, Xiangwei Geng, Ruoyi Zhou, Dingchen Yang, Bo Dai, Lewei Lu, Hongyang Li*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2303.10340)]


# Applications

## 2D Visual Perception
<a id="2DVisualPerception-link"></a>

### Image Classification
<a id="Classification-link"></a>
**Is synthetic data from generative models ready for image recognition?**  
*Ruifei He, Shuyang Sun, Xin Yu, Chuhui Xue, Wenqing Zhang, Philip Torr, Song Bai, Xiaojuan Qi*  
ICLR 2023 [[Paper](https://arxiv.org/abs/2206.09592)][[Code](https://github.com/CVMI-Lab/SyntheticData)]

**Synthetic Data from Diffusion Models Improves ImageNet Classification**  
*Shekoofeh Azizi, Simon Kornblith, Chitwan Saharia, Mohammad Norouzi, David J. Fleet*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2304.08466)]

**Adapting Pretrained Vision-Language Foundational Models to Medical Imaging Domains**  
*Pierre Chambon, Christian Bluethgen, Curtis P. Langlotz, Akshay Chaudhari*  
NeurIPS 2022 [[Paper](https://arxiv.org/abs/2210.04133)]

**OpenGAN: Open-Set Recognition via Open Data Generation**  
*Shu Kong, Deva Ramanan*  
ICCV 2021 [[Paper](https://arxiv.org/abs/2104.02939)][[Project](https://www.cs.cmu.edu/~shuk/OpenGAN.html)][[Code](https://github.com/aimerykong/OpenGAN)][[Video](https://www.youtube.com/watch?v=CNYqYXyUHn0)]

Image Captions are Natural Prompts for Text-to-Image Models

Diffusion Models and Semi-Supervised Learners Benefit Mutually with Few Labels

### Image Segmentation
<a id="Segmentation-link"></a>
**Learning Semantic Segmentation from Synthetic Data: A Geometrically Guided Input-Output Adaptation Approach**  
*Yuhua Chen, Wen Li, Xiaoran Chen, Luc Van Gool*  
CVPR 2019 [[Paper](https://arxiv.org/abs/1812.05040)]

**Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization**  
*Daiqing Li, Junlin Yang, Karsten Kreis, Antonio Torralba, Sanja Fidler*  
CVPR 2021 [[Paper](https://arxiv.org/abs/2104.05833)][[Project](https://nv-tlabs.github.io/semanticGAN/)][[Code](https://github.com/nv-tlabs/semanticGAN_code)]

**Repurposing GANs for One-shot Semantic Part Segmentation**  
*Nontawat Tritrong, Pitchaporn Rewatbowornwong, Supasorn Suwajanakorn*  
CVPR 2021 [[Paper](https://arxiv.org/abs/2103.04379)][[Project](https://repurposegans.github.io/)][[Code](https://github.com/bryandlee/repurpose-gan/)]

**Diffusion Models for Zero-Shot Open-Vocabulary Segmentation**  
*Laurynas Karazija, Iro Laina, Andrea Vedaldi, Christian Rupprecht*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2306.09316)]

**DifFSS: Diffusion Model for Few-Shot Semantic Segmentation**  
*Weimin Tan, Siyuan Chen, Bo Yan*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2307.00773)]

Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization

Few-shot 3D Multi-modal Medical Image Segmentation using Generative Adversarial Learning

Segmentation in Style: Unsupervised Semantic Image Segmentation with Stylegan and CLIP

**BigDatasetGAN: Synthesizing ImageNet with Pixel-wise Annotations**  
*Daiqing Li, Huan Ling, Seung Wook Kim, Karsten Kreis, Adela Barriuso, Sanja Fidler, Antonio Torralba*  
CVPR 2022 [[Paper](https://arxiv.org/abs/2201.04684)][[Project](https://nv-tlabs.github.io/big-datasetgan/)][[Code](https://github.com/nv-tlabs/bigdatasetgan_code)]

**HandsOff: Labeled Dataset Generation With No Additional Human Annotations**  
*Austin Xu, Mariya I. Vasileva, Achal Dave, Arjun Seshadri*  
CVPR 2023 [[Paper](https://arxiv.org/abs/2212.12645)][[Project](https://austinxu87.github.io/handsoff/)][[Code](https://github.com/austinxu87/handsoff/)]

**Dataset Diffusion: Diffusion-based Synthetic Dataset Generation for Pixel-Level Semantic Segmentation**  
*Quang Nguyen, Truong Vu, Anh Tran, Khoi Nguyen*  
NeurIPS 2023 [[Paper](https://arxiv.org/abs/2309.14303)[[Code](https://github.com/VinAIResearch/Dataset-Diffusion)]

**MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation**  
*Jiahao Xie, Wei Li, Xiangtai Li, Ziwei Liu, Yew Soon Ong, Chen Change Loy*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2309.13042)][[Code](https://github.com/Jiahao000/MosaicFusion)]

**Label-Efficient Semantic Segmentation with Diffusion Models**  
*Dmitry Baranchuk, Ivan Rubachev, Andrey Voynov, Valentin Khrulkov, Artem Babenko*  
ICLR 2022 [[Paper](https://arxiv.org/abs/2112.03126)][[Project](https://yandex-research.github.io/ddpm-segmentation/)][[Code](https://github.com/yandex-research/ddpm-segmentation)]

**[ODISE] Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models**  
*Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, Shalini De Mello*  
CVPR 2023 [[Paper](https://arxiv.org/abs/2303.04803)][[Project](https://jerryxu.net/ODISE/)][[Code](https://github.com/NVlabs/ODISE)]


### Object Detection
<a id="Detection-link"></a>

**[GeoDiffusion] Integrating Geometric Control into Text-to-Image Diffusion Models for High-Quality Detection Data Generation via Text Prompt**  
*Kai Chen, Enze Xie, Zhe Chen, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2306.04607)][[Project](https://kaichen1998.github.io/projects/geodiffusion/)]

**Explore the Power of Synthetic Data on Few-shot Object Detection**  
*Shaobo Lin, Kun Wang, Xingyu Zeng, Rui Zhao*  
CVPRW 2023 [[Paper](https://arxiv.org/abs/2303.13221)]

**[SODGAN] Synthetic Data Supervised Salient Object Detection**  
*Zhenyu Wu, Lin Wang, Wei Wang, Tengfei Shi, Chenglizhao Chen, Aimin Hao, Shuo Li*  
ACM MM 2022 [[Paper](https://arxiv.org/abs/2210.13835)]

**ImaginaryNet: Learning Object Detectors without Real Images and Annotations**  
*Minheng Ni, Zitong Huang, Kailai Feng, Wangmeng Zuo*  
ICLR 2023 [[Paper](https://arxiv.org/abs/2210.06886)][[Code](https://github.com/kodenii/ImaginaryNet)]

**The Big Data Myth: Using Diffusion Models for Dataset Generation to Train Deep Detection Models**  
*Roy Voetman, Maya Aghaei, Klaas Dijkstra*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2306.09762)]


## Visual Generation
<a id="VisualGeneration-link"></a>
**Re-Aging GAN: Toward Personalized Face Age Transformation**  
*Farkhod Makhmudkhujaev, Sungeun Hong, and In Kyu Park*  
ICCV 2021 [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Makhmudkhujaev_Re-Aging_GAN_Toward_Personalized_Face_Age_Transformation_ICCV_2021_paper.pdf)][[Video](https://www.youtube.com/watch?v=NRl0GPgtcBY)]

**Only a Matter of Style: Age Transformation Using a Style-Based Regression Model**  
*Yuval Alaluf, Or Patashnik, Daniel Cohen-Or*  
SIGGRAPH 2021 [[Paper](https://arxiv.org/abs/2102.02754)][[Project](https://yuval-alaluf.github.io/SAM/)][[Code](https://github.com/yuval-alaluf/SAM)][[Video](https://www.youtube.com/watch?v=X_pYC_LtBFw)]

**Production-Ready Face Re-Aging for Visual Effects**  
*Gaspard Zoss, Prashanth Chandran, Eftychios Sifakis, Markus Gross, Paulo Gotardo, Derek Bradley*  
TOG 2021 [[Paper](https://studios.disneyresearch.com/app/uploads/2022/10/Production-Ready-Face-Re-Aging-for-Visual-Effects.pdf)][[Project](https://studios.disneyresearch.com/2022/11/30/production-ready-face-re-aging-for-visual-effects/)][[Video](https://www.youtube.com/watch?v=ZP1ApcdyAjk)]

**Zero-1-to-3: Zero-shot One Image to 3D Object**  
*Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, Carl Vondrick*  
ICCV 2023 [[Paper](https://arxiv.org/abs/2303.11328)][[Project](https://zero123.cs.columbia.edu/)][[Code](https://github.com/cvlab-columbia/zero123)]

**DreamBooth3D: Subject-Driven Text-to-3D Generation**  
*Amit Raj, Srinivas Kaza, Ben Poole, Michael Niemeyer, Nataniel Ruiz, Ben Mildenhall, Shiran Zada, Kfir Aberman, Michael Rubinstein, Jonathan Barron, Yuanzhen Li, Varun Jampani*   
arXiv 2023 [[Paper](https://arxiv.org/abs/2303.13508)][[Project](https://dreambooth3d.github.io/)][[Video](https://youtu.be/kKVDrbfvOoA)]

**StyleAvatar3D: Leveraging Image-Text Diffusion Models for High-Fidelity 3D Avatar Generation**  
*Chi Zhang, Yiwen Chen, Yijun Fu, Zhenglin Zhou, Gang YU, Billzb Wang, Bin Fu, Tao Chen, Guosheng Lin, Chunhua Shen*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2305.19012)]


## Self-supervised Learning
<a id="SelfsupervisedLearning-link"></a>
**Generative Models as a Data Source for Multiview Representation Learning**  
*Ali Jahanian, Xavier Puig, Yonglong Tian, Phillip Isola*  
ICLR 2022 [[Paper](https://arxiv.org/abs/2106.05258)][[Project](https://ali-design.github.io/GenRep/)][[Code](https://github.com/ali-design/GenRep)][[Video](https://www.youtube.com/watch?v=qYmGvVrGZno)]

**StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners**  
*Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, Dilip Krishnan*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2306.00984)]

**Ensembling with Deep Generative Views**  
*Lucy Chai, Jun-Yan Zhu, Eli Shechtman, Phillip Isola, Richard Zhang*  
CVPR 2021 [[Paper](https://arxiv.org/abs/2104.14551)][[Project](https://chail.github.io/gan-ensembling/)][[Code](https://github.com/chail/gan-ensembling)]

**DreamTeacher: Pretraining Image Backbones with Deep Generative Models**  
*Daiqing Li, Huan Ling, Amlan Kar, David Acuna, Seung Wook Kim, Karsten Kreis, Antonio Torralba, Sanja Fidler*  
ICCV 2023 [[Paper](https://arxiv.org/abs/2307.07487)][[Project](https://research.nvidia.com/labs/toronto-ai/DreamTeacher/)]


## 3D Visual Perception
<a id="3DVisualPerception-link"></a>

### Robotics
<a id="Robotics-link"></a>

**INeRF: Inverting Neural Radiance Fields for Pose Estimation**  
*Lin Yen-Chen, Pete Florence, Jonathan T. Barron, Alberto Rodriguez, Phillip Isola, Tsung-Yi Lin*  
IROS 2021 [[Paper](https://arxiv.org/abs/2012.05877)][[Project](https://yenchenlin.me/inerf/)][[Code](https://github.com/yenchenlin/iNeRF-public)][[Video](https://www.youtube.com/watch?v=eQuCZaQN0tI)]

**LENS: Localization enhanced by NeRF synthesis**  
*Arthur Moreau, Nathan Piasco, Dzmitry Tsishkou, Bogdan Stanciulescu, Arnaud de La Fortelle*  
CoRL 2021 [[Paper](https://arxiv.org/abs/2110.06558)][[Video](https://www.youtube.com/watch?v=DgIpVoS6ejY)]

**NeRF-Pose: A First-Reconstruct-Then-Regress Approach for Weakly-supervised 6D Object Pose Estimation**  
*Fu Li, Hao Yu, Ivan Shugurov, Benjamin Busam, Shaowu Yang, Slobodan Ilic*  
arXiv 2022 [[Paper](https://arxiv.org/abs/2203.04802)]

**Vision-only robot navigation in a neural radiance world**  
*Michal Adamkiewicz, Timothy Chen, Adam Caccavale, Rachel Gardner, Preston Culbertson, Jeannette Bohg, Mac Schwager*  
RAL 2022 [[Paper](https://arxiv.org/abs/2110.00168)][[Project](https://mikh3x4.github.io/nerf-navigation/)][[Code](https://github.com/mikh3x4/nerf-navigation)][[Video](https://www.youtube.com/watch?v=5JjWpv9BaaE)]

**Event-based Camera Tracker by ∇t NeRF**  
*Mana Masuda, Yusuke Sekikawa, Hideo Saito*  
WACV 2023 [[Paper](https://arxiv.org/abs/2304.04559)]


### Autonomous Driving
<a id="AutonomousDriving-link"></a>

**Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field**  
*Leheng Li, Qing Lian, Luozhou Wang, Ningning Ma, Ying-Cong Chen*  
CVPR 2023 [[Paper](https://arxiv.org/abs/2304.03526)][[Project](https://len-li.github.io/lift3d-web/)][[Code](https://github.com/EnVision-Research/Lift3D)]

**UniSim: A Neural Closed-Loop Sensor Simulator**  
*Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, Raquel Urtasun*  
CVPR 2023 [[Paper](https://arxiv.org/abs/2308.01898)][[Project](https://waabi.ai/unisim/)][[Video](https://research-assets.waabi.ai/wp-content/uploads/2023/05/UniSim-video_compressed.mp4)]

**MARS: An Instance-aware, Modular and Realistic Simulator for Autonomous Driving**  
*Zirui Wu, Tianyu Liu, Liyi Luo, Zhide Zhong, Jianteng Chen, Hongmin Xiao, Chao Hou, Haozhe Lou, Yuantao Chen, Runyi Yang, Yuxin Huang, Xiaoyu Ye, Zike Yan, Yongliang Shi, Yiyi Liao, Hao Zhao*  
CICAI 2023 [[Paper](https://arxiv.org/abs/2307.15058)][[Project](https://open-air-sun.github.io/mars/)][[Code](https://github.com/OPEN-AIR-SUN/mars)][[Video](https://www.youtube.com/watch?v=AdC-jglWvfU)]


## Other Applications
<a id="OtherApplications-link"></a>

### Medical
<a id="Medical-link"></a>

**[MedGAN] Generating Multi-label Discrete Patient Records using Generative Adversarial Networks**  
*Edward Choi, Siddharth Biswal, Bradley Malin, Jon Duke, Walter F. Stewart, Jimeng Sun*  
MLHC 2017 [[Paper](https://arxiv.org/abs/1703.06490)]

**CorGAN: Correlation-Capturing Convolutional Generative Adversarial Networks for Generating Synthetic Healthcare Records**  
*Amirsina Torfi, Edward A. Fox*  
FLAIRS 2020 [[Paper](https://arxiv.org/abs/2001.09346)][[Code](https://github.com/astorfi/cor-gan)]

**Skin Lesion Classification Using GAN based Data Augmentation**  
*Rashid Haroon, Tanveer M. Asjid, Aqeel Khan Hassan*  
EMBC 2019 [[Paper](https://ieeexplore.ieee.org/document/8857905)]

**GAN-based synthetic medical image augmentation for increased CNN performance in liver lesion classification**  
*Maayan Frid-Adar, Idit Diamant, Eyal Klang, Michal Amitai, Jacob Goldberger, Hayit Greenspan*  
Neurocomputing 2018 [[Paper](https://arxiv.org/abs/1803.01229)]

**Synthetic data augmentation using GAN for improved liver lesion classification**  
*Maayan Frid-Adar, Eyal Klang, Michal Amitai, Jacob Goldberger, Hayit Greenspan*  
ISBI 2018 [[Paper](https://arxiv.org/abs/1801.02385)]

**Leveraging GANs for data scarcity of COVID-19: Beyond the hype**  
*Hazrat Ali, Christer Gronlund, Zubair Shah*  
CVPRW 2023 [[Paper](https://arxiv.org/abs/2304.03536)]

### Testing Data
<a id="Test-link"></a>

**Benchmarking Deepart Detection**  
*Yabin Wang, Zhiwu Huang, Xiaopeng Hong*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2302.14475)]

**Benchmarking Robustness to Text-Guided Corruptions**  
*Mohammadreza Mofayezi, Yasamin Medghalchi*  
CVPRW 2023 [[Paper](https://arxiv.org/abs/2304.02963)][[Code](https://github.com/ckoorosh/RobuText)]


## Datasets
<a id="Datasets-link"></a>

DiffusionDB (https://github.com/poloclub/diffusiondb)  

JourneyDB (https://github.com/JourneyDB/JourneyDB)  

Pick-a-Pic v2 (https://huggingface.co/datasets/yuvalkirstain/pickapic_v2)  

ImageReward (https://huggingface.co/datasets/THUDM/ImageRewardDB)  

HPD v2 (https://huggingface.co/datasets/xswu/human_preference_dataset)  

DFFD (http://cvlab.cse.msu.edu/dffd-dataset.html)  

ForgeryNet (https://yinanhe.github.io/projects/forgerynet.html)  

CNNSpot (https://github.com/peterwang512/CNNDetection)  

CIFAKE (https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

GenImage (https://github.com/GenImage-Dataset/GenImage)  


## :love_you_gesture: Citation
If you find our work useful for your research, please consider citing the paper:
```bibtex
@article{yang2023aigs,
  title={AI-Generated Images as Data Source: The Dawn of Synthetic Era},
  author={Zuhao Yang and Fangneng Zhan and Kunhao Liu and Muyu Xu and Shijian Lu},
  journal={arXiv preprint arXiv:2310.01830},
  year={2023}
}
```
