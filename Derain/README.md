


## Deraining Experiments

### Installation

* (More detail See [environment.yml](environment.yml))
A suitable [conda](https://conda.io/) environment named `rcdtorch` can be created and activated with:

```
conda env create -f environment.yml
conda activate rcdtorch
```

### Dataset & Training & Testing
Please refer to the original coding framework [RCDNet](https://github.com/hongwang01/RCDNet) to execute the training and testing process.

Running examples are also provided in the [run.sh](run.sh) file.

### Use EQ-REG
The calculation of EQ-REG is in the [F_Conv_YL.py](F_Conv_YL.py) file.

For the modification of the model, please compare [rcdnet_loss_xnet.py](rcdnet_loss_xnet.py) and [rcdnet.py](rcdnet.py).

For the modification of the training code, please compare [main_loss.py](main_loss.py), [trainer_loss.py](trainer_loss.py) and [main.py](main.py), [trainer.py](trainer.py).

