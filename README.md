# AI-Generated Images as Data Source: The Dawn of Synthetic Era
This project is associated with our survey paper which comprehensively contextualizes the advance of the recent **AI**-**G**enerated Images as Data **S**ource (**AIGS**) and visual AIGC by formulating taxonomies according to methodologies and applications.

**AI-Generated Images as Data Source: The Dawn of Synthetic Era** [[Paper](https://www.overleaf.com/2721143181hysckdgbdyqd)]  
*Zuhao Yang, Fangneng Zhan, Kunhao Liu, Muyu Xu, Shijian Lu*  
arXiv, 2023 

You are welcome to promote papers via pull request.
The process to submit a pull request:
- a. Fork the project into your own repository.
- b. Add the Title, Author, Conference, Paper link, Project link, and Code link in `README.md` with below format:
>**Title**<br>
>*Author*<br>
>Conference  
>[[Paper](Paper link)][[Project](Project link)][[Code](Code link)][[Video](Video link)] 
- c. Submit the pull request to this branch.

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

**Fake It Till You Make It: Guidelines for Effective Synthetic Data Generation**  
*Fida K. Dankar, Mahmoud Ibrahim*  
Applied Sciences 2021 [[Paper](https://www.mdpi.com/2076-3417/11/5/2158)]

## Table of Contents (Work in Progress)
Synthetic Data for Training:
- [Dataset Generation](#LabelledDataGeneration-link)
- [Data Augmentation](#DataAugmentation-link)

Synthetic Data Application:
- [Self-supervised Learning](#SelfsupervisedLearning-link)
- [Visual Understanding](#VisualUnderstanding-link)
  - [Classification](#Classification-link)
  - [Segmentation](#Segmentation-link)
  - [Detection](#Detection-link)
- [Visual Generation](#VisualGeneration-link)
- [Robotics](#Robotics-link)

Datasets:
- [Datasets](#Datasets-link)

## Dataset Generation
<a id="LabelledDataGeneration-link"></a>
**DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort**  
*Yuxuan Zhang, Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche, Adela Barriuso, Antonio Torralba, Sanja Fidler*   
CVPR 2021 [[Paper](https://arxiv.org/abs/2104.06490)][[Project](https://nv-tlabs.github.io/datasetGAN/)][[Code](https://github.com/nv-tlabs/datasetGAN_release/tree/master)]

**BigDatasetGAN: Synthesizing ImageNet with Pixel-wise Annotations**  
*Daiqing Li, Huan Ling, Seung Wook Kim, Karsten Kreis, Adela Barriuso, Sanja Fidler, Antonio Torralba*  
CVPR 2022 [[Paper](https://arxiv.org/abs/2201.04684)][[Project](https://nv-tlabs.github.io/big-datasetgan/)][[Code](https://github.com/nv-tlabs/bigdatasetgan_code)]

HandsOff: Labeled Dataset Generation With No Additional Human Annotations

On the generation of realistic synthetic petrographic datasets using a style-based GAN

Learning to Annotate Part Segmentation with Gradient Matching

Histopathology DatasetGAN: Synthesizing Large-Resolution Histopathology Datasets

Application of DatasetGAN in medical imaging: preliminary studies

Medical Image Segmentation Using Deep Learning: A Survey

Transformation-Consistent Self-Ensembling Model for Semisupervised Medical Image Segmentation

A Data Augmentation Pipeline to Generate Synthetic Labeled Datasets of 3D Echocardiography Images Using a GAN

## Data Augmentation
<a id="DataAugmentation-link"></a>
**A data augmentation perspective on diffusion models and retrieval**  
*Max F. Burg, Florian Wenzel, Dominik Zietlow, Max Horn, Osama Makansi, Francesco Locatello, Chris Russell*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2304.10253)]

**Effective Data Augmentation With Diffusion Models**  
*Brandon Trabucco, Kyle Doherty, Max Gurinas, Ruslan Salakhutdinov*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2302.07944)][[Project](http://btrabuc.co/da-fusion/)]

**Skin Lesion Classification Using GAN based Data Augmentation**  
*Rashid Haroon, Tanveer M. Asjid, Aqeel Khan Hassan*  
EMBC 2019 [[Paper](https://ieeexplore.ieee.org/document/8857905)]

GAN-based synthetic medical image augmentation for increased CNN performance in liver lesion classification

Data augmentation generative adversarial networks

Gan augmentation: Augmenting training data using generative adversarial networks

Enhancement of Image Classification Using Transfer Learning and GAN-Based Synthetic Data Augmentation

Diversify your vision datasets with automatic diffusion-based augmentation

## Self-supervised Learning
<a id="SelfsupervisedLearning-link"></a>
**Generative Models as a Data Source for Multiview Representation Learning**  
*Ali Jahanian, Xavier Puig, Yonglong Tian, Phillip Isola*  
ICLR 2022 [[Paper](https://arxiv.org/abs/2106.05258)][[Project](https://ali-design.github.io/GenRep/)][[Code](https://github.com/ali-design/GenRep)][[Video](https://www.youtube.com/watch?v=qYmGvVrGZno)]

**StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners**  
*Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, Dilip Krishnan*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2306.00984)]

## Visual Understanding
<a id="VisualUnderstanding-link"></a>

### Classification
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

### Segmentation
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

ODISE

### Detection
<a id="Detection-link"></a>
**DALL-E for Detection: Language-driven Compositional Image Synthesis for Object Detection**  
*Yunhao Ge, Jiashu Xu, Brian Nlong Zhao, Neel Joshi, Laurent Itti, Vibhav Vineet*  
arXiv 2022 [[Paper](https://arxiv.org/abs/2206.09592)]

**Explore the Power of Synthetic Data on Few-shot Object Detection**  
*Shaobo Lin, Kun Wang, Xingyu Zeng, Rui Zhao*   
CVPR 2023 [[Paper](https://arxiv.org/abs/2303.13221)]

**The Big Data Myth: Using Diffusion Models for Dataset Generation to Train Deep Detection Models**  
*Roy Voetman, Maya Aghaei, Klaas Dijkstra*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2306.09762)]

IMAGINARYNET: LEARNING OBJECT DETECTORS WITHOUT REAL IMAGES AND ANNOTATIONS

Integrating Geometric Control into Text-to-Image Diffusion Models for High-Quality Detection Data Generation via Text Prompt

## Visual Generation
<a id="VisualGeneration-link"></a>
DreamBooth3D  
StyleAvatar3D
Zero-1-to-3

## Robotics
<a id="Robotics-link"></a>
**NeRF-Supervision: Learning Dense Object Descriptors from Neural Radiance Fields**  
*Thomas Lips, Victor-Louis De Gusseme, Francis wyffels*  
ICRA 2022 [[Paper](https://arxiv.org/abs/2203.01913)][[Project](https://yenchenlin.me/nerf-supervision/)][[Code](https://github.com/yenchenlin/nerf-supervision-public)][[Video](https://www.youtube.com/watch?v=_zN-wVwPH1s)]

## Datasets
<a id="Datasets-link"></a>
**GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image**  
*Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, Yunhe Wang*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2306.08571)][[Project](https://genimage-dataset.github.io/)][[Code](https://github.com/GenImage-Dataset/GenImage)]

**DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models**  
*Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, Duen Horng Chau*  
ACL 2023 [[Paper](https://arxiv.org/abs/2210.14896)][Project](https://poloclub.github.io/diffusiondb/)][[Code](https://github.com/poloclub/diffusiondb)]

**JourneyDB: A Benchmark for Generative Image Understanding**  
*Junting Pan, Keqiang Sun, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, Hongsheng Li*  
arXiv 2023 [[Paper](https://arxiv.org/abs/2307.00716)][[Project](https://journeydb.github.io/)][[Code](https://github.com/JourneyDB/JourneyDB)]

GenImage (https://pan.baidu.com/share/init?surl=i0OFqYN5i6oFAxeK6bIwRQ#list/path=%2F)  
DiffusionDB (https://huggingface.co/datasets/poloclub/diffusiondb)  
JourneyDB (https://docs.google.com/forms/d/e/1FAIpQLSeiciK0g0IA46_hFaitRhdpihhpjqt3helJNT68y-C8MfKhiQ/viewform?pli=1)  
DreamBooth (https://github.com/google/dreambooth)  
BigDatasetGAN (https://drive.google.com/drive/u/2/folders/1NC0VSZrVZsd1Z_IRSdadyfCfTXMAPsf1)
