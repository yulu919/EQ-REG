


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

Running examples are also provided in the [run.sh](./RCDNet_code/for_syn/src/run.sh) file.

### Use EQ-REG
The calculation of EQ-REG is in the [F_Conv_YL.py](./RCDNet_code/for_syn/src/model/F_Conv_YL.py) file.

For the modification of the model, please compare [rcdnet_loss_xnet.py](./RCDNet_code/for_syn/src/model/rcdnet_loss_xnet.py) and [rcdnet.py](./RCDNet_code/for_syn/src/model/rcdnet.py).

For the modification of the training code, please compare [main_loss.py](./RCDNet_code/for_syn/src/main_loss.py), [trainer_loss.py](./RCDNet_code/for_syn/src/trainer_loss.py) and [main.py](./RCDNet_code/for_syn/src/main.py), [trainer.py](./RCDNet_code/for_syn/src/trainer.py).

