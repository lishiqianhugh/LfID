# Learning from intuition on PHYRE
This is an implementation of Vision Transformer, Swin Transformer and BEiT on [PHYRE](https://phyre.ai/).
## How to use
* Environment
  * The project is developed and tested with python 3.6, pytorch 1.1 and cuda 11.6, but any version newer than that should work.
  * To get the pretrained models, you may want to install `timm` from Tsinghua open source using
  ```
  pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
  For simple installation of the packages needed, please run `requirements.sh` in `LfI`.
* Prepare dataset
  * Customize your train dataloader by modify `configs/phyre_cfg.py ` and run dataset/phyreo.py to get train dataloader for faster training.
  ```
  python dataset/phyreo.py
  ```
  * Or directly call the PHYREO class in an online manner in `train.py`.


* Train LfI models
  * There are three candidates and two types of LfI models to train (Pretrained / From scratch).
  * Run `train.py` to train specified model on the protocal and fold you want, such as
  ```
  python train.py --model_name=ViT --protocal=within --fold=0 --batch_size=128
  ```
  
  * The log and model parameters will be saved in `{args.model_name}_{args.protocal}{args.fold}/`
* Evaluate
  * Run evaluate.py to test the model with specified protocal and fold
  ```
  python evaluate.py --model_path='ViT_within0/within0_10' --model_name=ViT --protocal=within --fold=0 --batch_size=256
  ```

The results in the cross-template setting hold larger variance than the ones in the within-template setting. The models in the cross-template setting converge earlier than those in the within-template setting.
