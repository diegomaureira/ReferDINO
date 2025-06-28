<div align="center">

<h2>ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations</h2>

[Tianming Liang](https://tmliang.github.io/)¹  &emsp;
[Kun-Yu Lin](https://kunyulin.github.io/)¹  &emsp;
[Chaolei Tan](https://chaoleitan.github.io/)¹  &emsp;
[Jianguo Zhang](https://faculty.sustech.edu.cn/zhangjg/en/)² &emsp;
[Wei-Shi Zheng](https://www.isee-ai.cn/~zhwshi/)¹  &emsp;
[Jian-Fang Hu](https://isee-ai.cn/~hujianfang/)¹*

¹Sun Yat-sen University &emsp;
²Southern University of Science and Technology

**ICCV 2025**

<h3 align="center">
  <a href="https://isee-laboratory.github.io/ReferDINO/" target='_blank'>Project Page</a> |
  <a href="https://arxiv.org/abs/2501.14607" target='_blank'>Paper</a>
</h3>

</div>

![visual](assets/visual.jpg)

## 📢 News
* `2025.6.28` Swin-B checkpoints are released.
* `2025.6.27` All training and inference code for ReferDINO is released.
* `2025.6.25` ReferDINO is accepted to ICCV 2025 ! 🎉 
* `2025.3.28` Our [ReferDINO-Plus](https://github.com/iSEE-Laboratory/ReferDINO-Plus), an ensemble approach of ReferDINO and SAM2, 
achieved the 2nd place in [PVUW](https://pvuw.github.io/) challenge RVOS Track at CVPR 2025! 🎉 See our [report](https://arxiv.org/pdf/2503.23509) for details!

## 👨‍💻 TODO
- [ ] Release demo code and online demo.
- [X] Release model weights.
- [X] Release training and inference code.

## 🔎 Framework
![model](assets/model.png)

## Enviroment Setup
We have tested our code in PyTorch 1.11 and 2.5.1, so either version should be compatible.

```
# Clone the repo
git clone https://github.com/iSEE-Laboratory/ReferDINO.git
cd ReferDINO

# [Optional] Create a clean Conda environment
conda create -n referdino python=3.10 -y
conda activate referdino

# Pytorch
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia

# MultiScaleDeformableAttention
cd models/GroundingDINO/ops
python setup.py build install
python test.py
cd ../../..

# Other dependencies
pip install -r requirements.txt 
```

Download pretrained GroundingDino as follows and put them in the diretory `pretrained`.
```
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

## Data Preparation
Please refer to [DATA.md](assets/DATA.md) for data preparation.

The directory struture is organized as follows.

```
ReferFormer/
├── configs
├── data
│   ├── coco
│   ├── a2d_sentences
│   ├── jhmdb_sentences
│   ├── mevis
│   ├── ref_davis
│   └── ref_youtube_vos
├── datasets
├── models
├── eval
├── tools
├── util
├── pretrained
├── ckpt
├── misc.py
├── pretrainer.py
├── trainer.py
└── main.py
```

## Get Stated
The following batch sizes are suitable for training on 48G GPUs. The results are saved in `output/{dataset}/{version}/`.

* Pretrain `Swin-T` on `coco` datasets with 8 GPUs. You can either specify the gpu indices with `--gids 0 1 2 3`. 

```
python main.py -c configs/coco_swint.yaml -rm pretrain -bs 12 -ng 6 --epochs 20 --version swint --eval_off
```

* Finetune on Refer-YouTube-VOS with the pretrained checkpoints.
```
python main.py -c configs/ytvos_swint.yaml -rm train -bs 2 -ng 8 --version swint -pw output/coco/swint/checkpoints/best.pth.tar --eval_off
```

* Inference on Refer-YouTube-VOS.
```
PYTHONPATH=. python eval/inference_ytvos.py -c configs/ytvos_swint.yaml -ng 6 -ckpt output/ref_youtube_vos/swint/checkpoints/best.pth.tar --version swint
```

* Inference for A2D-Sentences (or JHMDB-Sentences).
```
PYTHONPATH=. python main.py -c configs/a2d_swint.yaml -rm train -ng 8 --version swint -ckpt output/a2d_sentences/swint/checkpoints/best.pth.tar --eval
```

## Model Zoo
We have released the following model weights on [HuggingFace](https://huggingface.co/liangtm/referdino/tree/main). Please download and put them in the diretory `ckpt`.

| Train Set             |   Backbone    | Checkpoint                                                                          |
|-----------------------|:-------------:|-------------------------------------------------------------------------------------|
| coco                  |    Swin-B     | [coco_swinb.pth](https://huggingface.co/liangtm/referdino/blob/main/coco_swinb.pth) |
| coco, ref-youtube-vos |    Swin-B     | [ryt_swinb.pth](https://huggingface.co/liangtm/referdino/blob/main/ryt_swinb.pth)   |
| coco, a2d-sentences   |    Swin-B     | [a2d_swinb.pth](https://huggingface.co/liangtm/referdino/blob/main/a2d_swinb.pth)   |
| mevis                 |    Swin-B     | [mevis_swinb.pth](https://huggingface.co/liangtm/referdino/blob/main/mevis_swinb.pth) |

## Acknowledgements
Our code is built upon [ReferFormer](https://github.com/wjn922/ReferFormer), [SOC](https://github.com/RobertLuo1/NeurIPS2023_SOC), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO). We sincerely appreciate these efforts.

## Citation
If you find our work helpful for your research, please consider citing our paper.
```bibtex
@inproceedings{liang2025referdino,
    title={ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations},
    author={Liang, Tianming and Lin, Kun-Yu and Tan, Chaolei and Zhang, Jianguo and Zheng, Wei-Shi and Hu, Jian-Fang},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    year={2025}
}
```
