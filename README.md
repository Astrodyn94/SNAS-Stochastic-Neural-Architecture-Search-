# SNAS(Stochastic Neural Architecture Search)
Pytorch implementation of SNAS (Caution : This is not official version and was not written by the author of the paper)

## Requirements
```
Python >= 3.6.5, PyTorch == 0.1
```

## Datasets
Cifar-10 datasets were used, (5000 for training / 5000 for validation).
Note that the authors of the paper used 25000 images for training and validation set, respectively.

##Hyperparaeters
Overall, I followed hyperparameters that were given in the paper.
However, there were several parameters that were not given in the paper.
Ex) Softmax Temperature ($\lambda$)

## Search Validation Accuracy
![1](./Search_Validation.png)
Search Valication Accuracy. 
