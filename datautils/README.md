# BinCoFer datasets

Considering that the BCSD model we use is trained on dataset from ArchLinux, we chose to compile programs from ArchLinux to allow the BCSD model to detect on a similarly distributed dataset. 

We first scraped all the packages in the ArchLinux system as of August 2023, totaling 11,369 projects. Then, by manually modifying the PKGBUILD documents, we added static linking compilation options and adjusted the compilation order to change dynamic linking to static linking. Ultimately, we successfully manually compiled 148 binary programs.

We divided the dataset using an 7:3 ratio, with 70%(110 binaries) used to select the threshold (threshold selection dataset) and 30% (38 binaries) used to experimentally validate the effect under the selected threshold (validation dataset).


archlinux_ground_truth_train1.csv shows the training set of archlinux, archlinux_ground_truth_test1.csv shows the test set of archlinux