# Two data augmentation methods for roubst retinal fundus images analysis

======================================================

## Introduction

### This repository has the source code for the paper "Robust retinal vessel segmentation from a data augmentation perspective" .


- **Channel-wise random gamma correction**

![channel wise random gamma correction](img/CWRGC.gif)

- **Channel-wise random vessel augmentation**

![channel wise random gamma correction](img/CWRVA.gif)

We provide the implementation of the proposed data augmentation methods, namely, channel-wise random gamma correction and channel-wise random vessel augmentation. You may run the **data\_augmentation\_demo** notebook to see how these two methods work. 

Since the vessel segmentation model could be easily trained based on the [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) library, here we only show how the exported **PaddlePaddle** inference model (trained on the DRIVE training set using the proposed data augmentation methods) could be used for segmenting real-world retinal fundus images. Interested readers could refer to the **vessel\_segmentation\_demo.py** file for details.

## Configuration

This code has been tested on python3.6 environment with the folloing site-packages:

    matplotlib==2.2.2
    numpy==1.16.4
    opencv-python==3.4.3
    paddlepaddle-gpu==1.6.2
    
To execute the vessel segmentation demo, you should download the inference model weights [here](https://pan.baidu.com/s/1PrA0KyYNamfm9UrDO3nLVA) (fetch code: 63kp), and then put them into the pdl_assets directory.
After that, run the following script:

    python vessel_segmentation_demo.py
    
This demo takes the domo.jpg (a compressed image from the DRIVE test set) as input and generates an image with the extracted vessel mask.

## Citation request

If you find this code useful in your research, please consider citing [our paper](https://link.springer.com/chapter/10.1007%2F978-3-030-87000-3_20).

    Sun X. et al. (2021) Robust Retinal Vessel Segmentation from a Data Augmentation Perspective. In: Fu H., Garvin M.K., MacGillivray T., Xu Y., Zheng Y. (eds) Ophthalmic Medical Image Analysis. OMIA 2021. Lecture Notes in Computer Science, vol 12970. Springer, Cham. https://doi.org/10.1007/978-3-030-87000-3_20 

Here is a BiBTex citation as well:

    @InProceedings{10.1007/978-3-030-87000-3_20,
        author="Sun, Xu
        and Fang, Huihui
        and Yang, Yehui
        and Zhu, Dongwei
        and Wang, Lei
        and Liu, Junwei
        and Xu, Yanwu",
        editor="Fu, Huazhu
        and Garvin, Mona K.
        and MacGillivray, Tom
        and Xu, Yanwu
        and Zheng, Yalin",
        title="Robust Retinal Vessel Segmentation from a Data Augmentation Perspective",
        booktitle="Ophthalmic Medical Image Analysis",
        year="2021",
        publisher="Springer International Publishing",
        address="Cham",
        pages="189--198",
        isbn="978-3-030-87000-3"
       }




