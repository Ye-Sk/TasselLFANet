# TasselLFANet
<p align="center">
  <img src="https://github.com/Ye-Sk/MrMT/blob/master/LFANet_infer.png"/>
</p>  

TasselLFANet based on detection method for plant counting, implementation of paper :   
[___TasselLFANet：A Novel Lightweight Multi-Branch Feature Aggregation Neural Network for High-throughput Image-based Maize Tassels Detection and Counting___](https://www.frontiersin.org/articles/10.3389/fpls.2023.1158940/abstract)

# Main results
### Object Detection on MrMT dataset
|Model|FPS|P|R|F1|AP@.5|AP@.5:.95|
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|LFANet-HE|125|0.947|0.926|0.936|0.962|0.518|
|LFANet|77|0.946|0.942|0.944|0.968|0.546|
* Speed is tested on Nvidia Quadro P5000 GPU（16G）
### Object Counting on MrMT dataset
|Model|MAE|RMSE|MAPE|R²|
| :----: | :----: | :----: | :----: | :----: |
|LFANet-HE|2.70|3.76|14.3%|0.9751|0.9751|
|LFANet|1.80|2.68|9.2%|0.9903|0.9903|


# Installation
1. The code we implement is based on PyTorch 1.8 and Python 3.6, please refer to the file `requirements.txt` to configure the required environment.      
2. To convenient install the required environment dependencies, you can also use the following command look like this :     
~~~
$ pip install -r requirements.txt 
~~~

# Build your own dataset
**To train your own datasets on this framework, we recommend that :**  
* Annotate your data with the image annotation tool [LabelIMG](https://github.com/heartexlabs/labelImg) to generate `.txt` labels.   
* Refer to the `config/data.yaml` example to configure your own hyperparameters file. 
* Based on the `train.py` code example configure your own training parameters.

# Training
### Prepare Your Data
1. You can download the `MrMT` dataset from [___Baidu Drive (9.2GB)___](https://github.com/Ye-Sk/MrMT)
2. Move your dataset into the `data` folder, please follow the format look like this :
~~~
├── data
│   ├── images
│   │   ├── train
│   │   ├── valid
│   │   └── test
│   ├── labels
│   │   ├── train
│   │   ├── valid
│   │   └── test
~~~
* Run the following command to start training :  
~~~
$ python train.py --dataset config/data.yaml --batch-size 16 --workers 8
~~~
___For some reasons, our experiment haven't use a pretrained model, and we recommend 
that you pretrain if resources are adequate, the gains from this are considerable.___

# Evaluation
* Run the following command to evaluate the results :  
~~~
$ python eval.py --model LFANet.pt --dataset config/data.yaml --imgsz 640
~~~
# Inference
* Run the following command on a variety of sources :   
~~~
$ python infer.py --imgsz 640 --source config/images  # on image
~~~
~~~
$ python infer.py --imgsz 640 --source 0  # on webcam
~~~
# Citation
~~~
@article{ye2023TasselLFANet,  
  title={TasselLFANet: A Novel Lightweight Multi-Branch Feature Aggregation Neural Network for High-throughput Image-based Maize Tassels Detection and Counting},  
  author={Yu, Zhenghong and Ye, Jianxiong and Li, Cuina and Zhou, Huabing and Li, Xun}, 
  journal={Frontiers in Plant Science}, 
  volume={14},
  pages={1291-1307},
  year={2023},
  doi={10.3389/fpls.2023.1158940}
}
~~~
