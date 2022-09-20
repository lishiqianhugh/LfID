# Learning from dynamics on PHYRE

The code structure is similar to the one in LfI. See [LfI.md](LfID/LfI/LfI.md) as a guide.

But there are still some changes expected to be stated.

* Train TimeSformer with ground-truth dynamics from PHYRE's simulator, run `train_gd.py`.
* Train dynamics prediction model PredRNN, run `train_pred.py`.
* Train the whole LfD pipeline, run `train_ad.py`, specify serial optimization by setting the argument _pred_resumed_ as _True_.
