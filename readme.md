# Rethinking Graph Neural Networks for Anomaly Detection 

This is the official implementation for the following paper:

[Rethinking Graph Neural Networks for Anomaly Detection](https://proceedings.mlr.press/v162/tang22b.html)  
*Jianheng Tang, Jiajin Li, Ziqi Gao, Jia Li*  
ICML 2022

BWGNN has been integrated into [GADBench](https://github.com/squareRoot3/GADBench), a comprehensive benchmark for (semi-)supervised graph anomaly detection.


Dependencies
----------------------
- pytorch 1.9.0
- dgl 0.8.1
- sympy
- argparse
- sklearn


How to run
--------------------------------
The T-Finance and T-Social datasets developed in the paper are on [google drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing). Download and unzip all files in the `dataset` folder.

`plot.zip` in the above link is used to reproduce Figure 1 and 2 in our paper. You can unzip it and directly run the corresponding `.py` files.

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

Train BWGNN (homo) on T-Social (40%):
```
python main.py --dataset tsocial --train_ratio 0.4 --hid_dim 10 \
--order 5 --homo 1 --epoch 100 --run 1
```



If you use this package and find it useful, please cite our ICML paper using the following BibTeX. Thanks! :)

```
@InProceedings{tang2022rethinking,
  title = 	 {Rethinking Graph Neural Networks for Anomaly Detection},
  author =       {Tang, Jianheng and Li, Jiajin and Gao, Ziqi and Li, Jia},
  booktitle = 	 {International Conference on Machine Learning},
  year = 	 {2022},
}
```
You can find a more detailed BibTex or other citation formats [here](https://proceedings.mlr.press/v162/tang22b.html).
