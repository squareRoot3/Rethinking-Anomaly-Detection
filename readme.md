# Rethinking Graph Neural Networks for Anomaly Detection

This is a PyTorch implementation of 

> Rethinking Graph Neural Networks for Anomaly Detection


Dependencies
----------------------
- pytorch 1.9.0
- dgl 0.8.1
- sympy
- argparse
- sklearn

How to run
--------------------------------
The T-Finance and T-Social datasets developed in the paper are on [google drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing). Download and unzip it into `dataset`.

The Yelp and Amazon datasets will be automatically downloaded from the Internet. 

Train BWGNN (homo) on Amazon (40%): 
```
python main.py --dataset amazon --train_ratio 0.4 --hid_dim 64 \
--order 2 --homo 1 --epoch 100 --run 1
```
`amazon` can be replaced by other datasets: `yelp/tfinance/tsocial`

Train BWGNN (hetero) on Yelp (1%):
```
python main.py --dataset yelp --train_ratio 0.01 --hid_dim 64 \
--order 2 --homo 0 --epoch 100 --run 1
```
BWGNN (hetero) only supports Yelp and Amazon.

Train BWGNN (homo) on tsocial (0.01%):
```
python main.py --dataset tsocial --train_ratio 0.0001 --hid_dim 10 \
--order 5 --homo 1 --epoch 100 --run 1
```