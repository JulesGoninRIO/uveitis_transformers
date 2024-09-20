# Uveitis

This repository contains every script that can be used for Vasculis project. There are 2 folders Data and Classification.


## Prerequisites

It can be run on local (T/Studies/Vasculitis/code) and on SOIN (/users/ch_amiotv/Desktop/vasculitis).

Environment have requierement.txt and are already there in SOIN:

ssh ch_amiotv@soin-rtx2080
/data/soin/vasculitis/vasc/bin/activate 

or 

ssh retina-ai
/data/soin/vasculitis/A_100/bin/activate 

Note that I needed to changed open-cv version to create the environment in SOIN, some other libraries are not the latest version to work in SOIN, but any working version of library should be ok

## Data/Annotations

Run only Data/Annotations/run.py

Get.py is used to dowload annotations from Discovery (HOJG/SOIN), using Cohort Builder (environment/server)
Cohort Builder is another project from our group.

Get.py create .csv annotation table for each annotator with every inflamatory signs.
table.csv is the main one from Uveitis expert Florence Hoogewoud with 543 eyes.
Other are in intergrader/ with at the 112 test set eyes annotated.

Split.py is used to split data into custom groups from a dictionary with {name: ratio}
It iteratively shuffle patients until eye annotation are balanced for each sign asked (as best).
For the paper it uses intergrader/Shalini_list.csv to start the test set and then balance as possible.

## Data/Metrics

Run Agreement.py for all intergrader statistics
Run Stats.py with correlation() or distribution() for custom statistics

OCI.py was manually implemented reading the cited paper ORDINAL CLASSIFICATION

table_error.py is used to highlight large errors from the model

Stats.py contains various functions to plot figures

Aggreement.py reads annotation tables, compute statistics and save plots using Stats.py

Folder contains output figures in table/ Teodora/ Muriel/ Shalini/ used for the paper

## Classification

Run main.py for custom experiment described in configs_DL.json
Run train_all.py to reproduce our experiments

models/ contaims iniatial swin-timm pretrained weights and final weights.
There is one final model for each inflamatory sign.

Data used for training is created by load_dataset.py from configs_DL.json and annotations

image_trainer.py (class from trainer.py) is the DL training script.
Scores and predictions are saved in results/

main.py also contains functions to store and compare several experiments.

gradcam.py uses function from pytorch_grad_cam/ (downloaded from public Git)
