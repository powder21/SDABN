# SDABN

Code of the paper [Synergy Between Semantic Segmentation and Image Denoising via Alternate Boosting](https://arxiv.org/abs/2102.12095)

## Prepare data
#### Cityscapes
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/). Your directory tree should be look like this: (data_root is specified by DATASET.ROOT in experiment/cityscapes/xxx.yaml)

````bash
$data_root
└── cityscapes
   ├── gtFine
   │   ├── test
   │   ├── train
   │   └── val
   └── leftImg8bit
       ├── test
       ├── train
       └── val
````

#### OutdoorSeg
You can download the [OutdoorSeg](https://1drv.ms/u/s!Ap2bi3TSun55lXmvV_7DmsJm3I1O?e=C5MYoF).

## Train
For one SDABN, train the models following the order of Seg<sub>1 </sub>--> Dn<sub>1 </sub>--> Seg<sub>2 </sub>--> Dn<sub>2 </sub>--> Seg<sub>3 </sub>--> Dn<sub>3 </sub>-->....

Example of training scipt on Cityscapes dataset
```
# train s1 and get s1.pth
cd cityscapes/seg/
python train.py --stage 1 --resume_s1 \path\to\seg_model\on\clean

# train d1 and get d1.pth
cd cityscapes/dn/
python train.py --stage 2 --resume_seg1 \path\to\s1.pth

# train s2 and get s2.pth
cd cityscapes/seg/
python train.py --stage 3 --resume_s1 \path\to\s1.pth --resume_d1 \path\to\d1.pth --resume_s2 \path\to\s1.pth

# train d2 and get d2.pth
cd cityscapes/dn/
python train.py --stage 4 --resume_seg1 \path\to\s1.pth --resume_d1 \path\to\d1.pth --resume_seg2 \path\to\s2.pth

# train s3 and get s3.pth
cd cityscapes/seg/
python train.py --stage 5 --resume_s1 \path\to\s1.pth --resume_d1 \path\to\d1.pth --resume_s2 \path\to\s2.pth --resume_d2 \path\to\d2.pth --resume_s3 \path\to\s2.pth

# train d3 and get d3.pth
cd cityscapes/dn/
python train.py --stage 6 --resume_seg1 \path\to\s1.pth --resume_d1 \path\to\d1.pth --resume_seg2 \path\to\s2.pth --resume_d2 \path\to\d2.pth --resume_seg3 \path\to\s3.pth
```

## Model link
[OneDrive](https://1drv.ms/u/s!Ap2bi3TSun55lXoRKSCLO95TTbr-?e=0skLwd)
In the link, sds.pth is the model of Seg<sub>2</sub>, and sdsdsd.pth is the model of Dn<sub>3.


