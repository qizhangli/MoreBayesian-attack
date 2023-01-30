# MoreBayesian-attack
Code for our ICLR 2023 paper [Making Substitute Models More Bayesian Can Enhance Transferability of Adversarial Examples](https://github.com/qizhangli/MoreBayesian-attack).

## Requirements
* Python 3.8.8
* PyTorch 1.12.0
* Torchvision 0.13.0

## Datasets
Select images from ImageNet validation set, and write ```.csv``` file as following:
```
class_index, class, image_name
0,n01440764,ILSVRC2012_val_00002138.JPEG
2,n01484850,ILSVRC2012_val_00004329.JPEG
...
```

## Finetune, Attack, and Evaluate
### Finetune
Perform our finetune with SWAG:
```
python3 finetune.py --data_path ${IMAGENET_DIR} --save-dir ${MODEL_SAVE_DIR}
```
### Attack
Perform attack:
```
python3 attack.py --source-model-dir ${SOURCE_MODEL_DIR} --data-dir ${IMAGENET_VAL_DIR} --data-info-dir ${DATASET_CSV_FILE} --save-dir ${ADV_IMG_SAVE_DIR}
```
### Evaluate
Evaluate the success rate of adversarial examples:
```
python3 test.py --dir ${ADV_IMG_SAVE_DIR} --model_dir ${VICTIM_MODEL_WEIGHTS_DIR}
```
## Acknowledgements
The following resources are very helpful for our work:

* [Pretrained models for ImageNet](https://github.com/Cadene/pretrained-models.pytorch)

## Citation
Please cite our work in your publications if it helps your research:

```
@article{li2023making,
  title={Making Substitute Models More Bayesian Can Enhance Transferability of Adversarial Examples},
  author={Li, Qizhang and Guo, Yiwen and Zuo, Wangmeng and Chen, Hao},
  booktitle={ICLR},
  year={2023}
}
```
