This repository contains the implementation of our paper

```
Nguyen, N. P., White, T., & Bunyak, F. Mitochondria Instance Segmentation in Electron Microscopy Image Volumes using 3D Deep Learning Networks (2021). IEEE/Applied Imagery Pattern Recognition Workshop, AIPR.
```

![3D_CLSTM](media/mito_pipeline.jpg "3D CLSTM")





It relies on the following projects:  
[CFCM-2D](https://github.com/faustomilletari/CFCM-2D)  
[ConvLSTM](https://github.com/rogertrullo/pytorch_convlstm)  
[MONAI](https://github.com/Project-MONAI/MONAI)  
[TorchIO](https://github.com/fepegar/torchio)  
[Segmenation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch)  
[Connectomics](https://github.com/zudi-lin/pytorch_connectomics)



## Install
```
pip install git+https://github.com/zudi-lin/pytorch_connectomics
```


```

pip install -r requirements.txt
```



## Train
```
python train_mito.py --model --data
```

with  
--model: check point path (default is None)
--data: data path   



## Test
```
python test_mito.py --model --data
```

with  
--model: check point path  
--data: data path








