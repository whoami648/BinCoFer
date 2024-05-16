# BinCoFer
BinCoFer is dedicated to detecting the partial reuse of third-party libraries at the binary level in C/C++ binary programs.
This project aims to collect Arch Linux data sets, mainly covering ground truth, binary programs and third-party libraries. However, these resources have not yet been fully curated.

# jtrans finetune
From the figure below, we can see that the model after tuning has improved by a certain amount compared with the model before tuning.If you want to get the **tuned model**, please contact with zyx72038@hust.edu.cn.

                                                same function pairs similarity
![jtrans_xiangsi](https://github.com/whoami648/BinCoFer/assets/75363525/3c2d543c-b599-48f6-80af-6b9419955ff6)

                                                different function pairs similarity
![jtrans_not_xiangsi](https://github.com/whoami648/BinCoFer/assets/75363525/b3de5582-5163-4be0-8898-d3cfa0b13017)



# Get Started
-----------------------------
# Prerequisites
- Linux, Windows
- Python 3.85
- PyTorch 2.2.0
- CUDA 12.2
- IDA pro 7.5

-----------------------------

# Quick Start
a. Create a conda virtual environment and activate it.

```conda create --name BinCoFer_env python=3.85 pandas tqdm -y```
`conda activate BinCoFer_env`

b. Install PyTorch and other packages.

`conda install pytorch cudatoolkit=12.2 -c pytorch`
`conda install --file requirements.txt`

c. Get code and dataset of BinCoFer.

`git clone https://github.com/whoami648/BinCoFer.git`


d. Calculate precision and recall

`python Percision_Recall.py -h`


-----------------------------

# Dataset
Considering that the BCSD model we use is trained on dataset from ArchLinux, we chose to compile programs from ArchLinux to allow the BCSD model to detect on a similarly distributed dataset. 

We first scraped all the packages in the ArchLinux system as of August 2023, totaling 11,369 projects. Then, by manually modifying the PKGBUILD documents, we added static linking compilation options and adjusted the compilation order to change dynamic linking to static linking. Ultimately, we successfully manually compiled 148 binary programs.

We divided the dataset using an 7:3 ratio, with 70%(110 binaries) used to select the threshold (threshold selection dataset) and 30% (38 binaries) used to experimentally validate the effect under the selected threshold (validation dataset).

