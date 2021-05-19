# Data Pre-processing Scipts.

This folder contains all the scripts used to pre-process our datasets before training.

## Table of contents
* [Dataset Partitioning](#data-partition)
* [Convert *.xml to *.csv](#xml_to_csv)
* [Convert *.xml to *.record](#xml_to_tf-records)


## Dataset Partitioning

Script: data_partitioning.py

This script is used to split the dataset into training, validation and testing sets.

We first split the dataset into training and testing sets in a ratio of 80:20. Before splitting is done, copy all training images, together with their corresponding
*.xml annotation files a single folder. 

And then after 20% of the training data was set aside as validation set.

usage: 
```
dataset_partitioning_script.py -x -i [PATH_TO_IMAGES_FOLDER] -r [x] 
```
where;

- PATH_TO_IMAGES_FOLDER -- represents the directory path where images and the XML annotation files are stored.
Still training and testing sets are created in the same directory.
- x -- the splitting value, for example if x = 0.2, then the training set is 80% of the whole dataset and testing set 20%.

For more info: https://github.com/sglvladi/TensorFlowObjectDetectionTutorial


## Convert *.xml to *.csv

Script: xml_to_csv.py
 
The script helps in generating csv files from the XML files. 

Usage: 
Place training, validation, and testing datasets folders into a single folder named "image" but the folder's name can be changed if at all you change lines 30 and 32 in the script to change name of that folder.

Then run
```
python xml_to_csv.py install
```
## Convert *.xml to *.record

Script: generate_tfrecord.py

This script helps in converting xml files to tf-records format


usage:-
Creating train set - tf-record file:
```
python generate_tfrecord.py -x [PATH_TO_TRAIN-SET_FOLDER]/train_set -l [PATH_TO_label_map_FOLDER]/label_map.pbtxt -o [PATH_TO_TF]/train_set.record
```
Creating validation set - tf-record file:
```
python generate_tfrecord.py -x [PATH_TO_VALIDATION-SET_FOLDER]/validation_set -l [PATH_TO_label_map_FOLDER]/label_map.pbtxt -o [PATH_TO_TF]/validation_set.record
```
where
 |PATH                          | MEANING                                                  |
 |----------------------------  |----------------------------------------------------------|
 | PATH_TO_TRAIN-SET_FOLDER     |is where the training set data is located.                |
 | PATH_TO_label_map_FOLDER is  | the location to the label map *.pbtxt script             |
 |PATH_TO_TF                    | path where you want to store the generated *.record files|
 | PATH_TO_VALIDATION-SET_FOLDER| is directory path to the validation data set.            |

This generates train_set.record and validation_set.record files.


For more info: This site was built using [TensorFlow-Object-Detection-API-Tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)