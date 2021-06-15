# Temporal Aggregate Representations for Long-Range Video Understanding

This repository provides official PyTorch implementation for our papers:

F. Sener, D. Singhania and A. Yao, "**Temporal Aggregate Representations for Long-Range Video Understanding**", ECCV 2020 [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610154.pdf)]

F. Sener, D. Chatterjee and A. Yao, "**Technical Report: Temporal Aggregate Representations**", arXiv:2106.03152, 2021 [[paper](https://arxiv.org/pdf/2106.03152.pdf)]


If you use the code/models hosted in this repository, please cite the following papers:

```
@inproceedings{sener2020temporal,
  title={Temporal aggregate representations for long-range video understanding},
  author={Sener, Fadime and Singhania, Dipika and Yao, Angela},
  booktitle={European Conference on Computer Vision},
  pages={154--171},
  year={2020},
  organization={Springer}
}
```

```
@article{sener2021technical,
  title={Technical Report: Temporal Aggregate Representations},
  author={Sener, Fadime and Chatterjee, Dibyadip and Yao, Angela},
  journal={arXiv preprint arXiv:2106.03152},
  year={2021}
}
```

## Dependencies
* Python3
* PyTorch
* Numpy, Pandas, PIL
* lmdb, tqdm

## Overview

This repository provides code to train, validate and test our models on the [EPIC-KITCHENS-55](https://openaccess.thecvf.com/content_ECCV_2018/papers/Dima_Damen_Scaling_Egocentric_Vision_ECCV_2018_paper.pdf) an [EPIC-KITCHENS-100](https://arxiv.org/pdf/2006.13256.pdf) datasets for the tasks of action anticipation and action recognition.

### Features

Follow the [RU-LSTM](https://github.com/fpv-iplab/rulstm) repository to download the RGB, Flow, Obj features and the train/val/test splits and keep them in the `data/ek55` or `data/ek100` folder depending on the dataset. Download the ROI features for EPIC-KITCHENS-100 from this link.

### Pretrained Models

Pretrained models are available only for the EPIC-KITCHENS-100 dataset trained on it's train split. They are provided in the folders `models_anticipation` and `model_recognition`.

### Validation

To validate our model, run the following:

#### EPIC-KITCHENS-55
##### Action Anticipation
* RGB: `python main_anticipation.py --mode validate --path_to_data data/ek55  --path_to_models models_anticipation/ek55 --modality rgb --video_feat_dim 1024`
* Flow: `python main_anticipation.py --mode validate --path_to_data data/ek55  --path_to_models models_anticipation/ek55 --modality flow --video_feat_dim 1024`
* Obj: `python main_anticipation.py --mode validate --path_to_data data/ek55  --path_to_models models_anticipation/ek55 --modality obj --video_feat_dim 352`
* ROI: `python main_anticipation.py --mode validate --path_to_data data/ek55  --path_to_models models_anticipation/ek55 --modality roi --video_feat_dim 1024`
* Late Fusion: `python main_anticipation.py --mode validate --path_to_data data/ek55  --path_to_models models_anticipation/ek55 --modality late_fusion`

##### Action Recognition
* RGB: `python main_recognition.py --mode validate --path_to_data data/ek55  --path_to_models models_recognition/ek55 --modality rgb --video_feat_dim 1024`
* Flow: `python main_recognition.py --mode validate --path_to_data data/ek55  --path_to_models models_recognition/ek55 --modality flow --video_feat_dim 1024`
* Obj: `python main_recognition.py --mode validate --path_to_data data/ek55  --path_to_models models_recognition/ek55 --modality obj --video_feat_dim 352`
* ROI: `python main_recognition.py --mode validate --path_to_data data/ek55  --path_to_models models_recognition/ek55 --modality roi --video_feat_dim 1024`
* Late Fusion: `python main_recognition.py --mode validate --path_to_data data/ek55  --path_to_models models_recognition/ek55 --modality late_fusion`

#### EPIC-KITCHENS-100
##### Action Anticipation
* RGB: `python main_anticipation.py --mode validate --ek100 --path_to_data data/ek100  --path_to_models models_anticipation/ek100/ --modality rgb --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`
* Flow: `python main_anticipation.py --mode validate --ek100 --path_to_data data/ek100  --path_to_models models_anticipation/ek100/ --modality flow --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`
* Obj: `python main_anticipation.py --mode validate --ek100 --path_to_data data/ek100  --path_to_models models_anticipation/ek100/ --modality obj --video_feat_dim 352 --num_class 3806 --verb_class 97 --noun_class 300`
* ROI: `python main_anticipation.py --mode validate --ek100 --path_to_data data/ek100  --path_to_models models_anticipation/ek100/ --modality roi --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`
* Late Fusion: `python main_anticipation.py --mode validate --ek100 --path_to_data data/ek100  --path_to_models models_anticipation/ek100/ --modality late_fusion --num_class 3806 --verb_class 97 --noun_class 300`

##### Action Recognition
* RGB: `python main_recognition.py --mode validate --ek100 --path_to_data data/ek100  --path_to_models models_recognition/ek100/ --modality rgb --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`
* Flow: `python main_recognition.py --mode validate --ek100 --path_to_data data/ek100  --path_to_models models_recognition/ek100/ --modality flow --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`
* Obj: `python main_recognition.py --mode validate --ek100 --path_to_data data/ek100  --path_to_models models_recognition/ek100/ --modality obj --video_feat_dim 352 --num_class 3806 --verb_class 97 --noun_class 300`
* ROI: `python main_recognition.py --mode validate --ek100 --path_to_data data/ek100  --path_to_models models_recognition/ek100/ --modality roi --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`
* Late Fusion: `python main_recognition.py --mode validate --ek100 --path_to_data data/ek100  --path_to_models models_recognition/ek100/ --modality late_fusion --num_class 3806 --verb_class 97 --noun_class 300`


Here are the validation results on EPIC-KITCHENS-100 as provided in our paper.

* Anticipation
![ant](https://drive.google.com/uc?export=view&id=1LASa8vtn9qQ0WGQdZcDEtTXOPdYdL11u)

* Recognition
![rec](https://drive.google.com/uc?export=view&id=1FJxOL3gXRRFr7g1cHhX0ZvCQiIaZCCRb)

### Testing and submitting the results to the server

To test your model on the EPIC-100 test split, run the following:
##### Action Anticipation
* `mkdir -p jsons/anticipation`
* `python main_anticipation.py --mode test --json_directory jsons/anticipation --ek100 --path_to_data data/ek100  --path_to_models models_anticipation/ek100/ --modality late_fusion --num_class 3806 --verb_class 97 --noun_class 300`

##### Action Recognition
* `mkdir -p jsons/recognition`
* `python main_recognition.py --mode test --json_directory jsons/recognition--ek100 --path_to_data data/ek100  --path_to_models models_recognition/ek100/ --modality late_fusion --num_class 3806 --verb_class 97 --noun_class 300`


### Custom Training

To train the model from scratch run the following (to fine-tune add --resume):

#### EPIC-KITCHENS-55
##### Action Anticipation
* RGB: `python main_anticipation.py --mode train --path_to_data data/ek55  --path_to_models models_anticipation/ek55 --modality rgb --video_feat_dim 1024`
* Flow: `python main_anticipation.py --mode train --path_to_data data/ek55  --path_to_models models_anticipation/ek55 --modality flow --video_feat_dim 1024`
* Obj: `python main_anticipation.py --mode train --path_to_data data/ek55  --path_to_models models_anticipation/ek55 --modality obj --video_feat_dim 352`
* ROI: `python main_anticipation.py --mode train --path_to_data data/ek55  --path_to_models models_anticipation/ek55 --modality roi --video_feat_dim 1024`

##### Action Recognition
* RGB: `python main_recognition.py --mode train --path_to_data data/ek55  --path_to_models models_recognition/ek55 --modality rgb --video_feat_dim 1024`
* Flow: `python main_recognition.py --mode train --path_to_data data/ek55  --path_to_models models_recognition/ek55 --modality flow --video_feat_dim 1024`
* Obj: `python main_recognition.py --mode train --path_to_data data/ek55  --path_to_models models_recognition/ek55 --modality obj --video_feat_dim 352`
* ROI: `python main_recognition.py --mode train --path_to_data data/ek55  --path_to_models models_recognition/ek55 --modality roi --video_feat_dim 1024`

#### EPIC-KITCHENS-100
##### Action Anticipation
* RGB: `python main_anticipation.py --mode train --ek100 --path_to_data data/ek100  --path_to_models models_anticipation/ek100/ --modality rgb --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`
* Flow: `python main_anticipation.py --mode train --ek100 --path_to_data data/ek100  --path_to_models models_anticipation/ek100/ --modality flow --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`
* Obj: `python main_anticipation.py --mode train --ek100 --path_to_data data/ek100  --path_to_models models_anticipation/ek100/ --modality obj --video_feat_dim 352 --num_class 3806 --verb_class 97 --noun_class 300`
* ROI: `python main_anticipation.py --mode train --ek100 --path_to_data data/ek100  --path_to_models models_anticipation/ek100/ --modality roi --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`

##### Action Recognition
* RGB: `python main_recognition.py --mode train --ek100 --path_to_data data/ek100  --path_to_models models_recognition/ek100/ --modality rgb --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`
* Flow: `python main_recognition.py --mode train --ek100 --path_to_data data/ek100  --path_to_models models_recognition/ek100/ --modality flow --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`
* Obj: `python main_recognition.py --mode train --ek100 --path_to_data data/ek100  --path_to_models models_recognition/ek100/ --modality obj --video_feat_dim 352 --num_class 3806 --verb_class 97 --noun_class 300`
* ROI: `python main_recognition.py --mode train --ek100 --path_to_data data/ek100  --path_to_models models_recognition/ek100/ --modality roi --video_feat_dim 1024 --num_class 3806 --verb_class 97 --noun_class 300`

Please refer to the papers for more technical details.

## Acknowledgements
This code is based on [RU-LSTM](https://github.com/fpv-iplab/rulstm), so thanks to the collaborators/maintainers of that repository.
