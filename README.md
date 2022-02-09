# BSB-Aerial-Dataset

This repository shows an example on how to use Detectron2's Panoptic-FPN in the BSB Aerial Dataset.

The dataset will be publicly available very shortly. For now, if you wish to have access to the data, please email me at osmarcarvalho@ieee.org.

The dataset is structured following the COCO annotation format recommendations.

For using the Detectron2 software, download it at https://github.com/facebookresearch/detectron2. Since we are using the image tiles in RGB format, please change the python file detection_utils for the one provided in this GitHub.

If you wish to build a new Remote Sensing Panoptic Segmentation dataset in the COCO annotation format, we suggest you reading the paper we have just published for instructions on how to build the dataset in GIS software such as ArcMap, and how to use our proposed software converter available (https://github.com/abilius-app/Panoptic-Generator).

If you are conducting similar studies, used the dataset, or have built another dataset using our proposed converter, please consider citing our paper.

If you have any questions or any trouble implementing any of these steps, please reach me out.
