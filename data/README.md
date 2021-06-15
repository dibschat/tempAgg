Follow the [RU-LSTM](https://github.com/fpv-iplab/rulstm) repository to download the RGB, Flow, Obj features and the train/val/test splits and keep them in the `data/ek55` or `data/ek100` folder depending on the dataset.

For ROI features we consider the union of the hand-object interaction bbox annotations provided by the authors of EPIC-KICTHENS-100 ([link](https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes)) as input and extract RGB features with TSN as explained [here](https://github.com/fpv-iplab/rulstm#feature-extraction).
