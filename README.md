# The Implementation of models for a thesis titled "Investigating automated bird counting from webcams using machine learning"
This repository hosts all the scripts used in the implementation of bird detection models. We are using Convolutional Neural Networks(CNN)'s Faster R-CNN, Single Shot Detector(SSD), and YOLOv3 meta-architectures while utilizing ResNet-101, MobileNet, Inception ResNet v2 and VGG-16 feature extraction Networks (backbone network).

We using MS COCO pre-trained models https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md and the Tensorflow Object detection API fine-tuned on our dataset. The data used in this study is collected from the live FeedWatcher cams of Cornell Lab Bird Cams suited in different parts of the United States. We used auto-screen software which captured approximately 1 Megapixel (Joint Photographic Experts Group)JPEG coloured images of resolution 1366x768 pixels from the feeds. At one of the stations in Treman bird feeding in Ithaca, New York, Axis P11448-LE camera is being used for recordings. A total of 10,592 images of different quality were collected and labelled manually using the LabelImg image annotation tool https://github.com/tzutalin/labelImg.

The repository provides all the files used to train and evaluate the models. But we have only attached a sample of our data since we couldn't upload the entire dataset of 10GB due to limited space allowed in the git free repository. A full dataset has been pubshiled on Zenodo for public access: https://zenodo.org/record/5172214#.YSO8YI4zZhH

# System Requirement 
We ran the experiments on an MSI GL75 Leopard 10SFR laptop with; 
1. CUDA 11.0, 
2. cuDNN SDK 8.0.4
3. Windows 10 x64
4. 10th Gen Intel Core i7-10750H,  
5. GeForce RTX 2070 8GB GDDR6 graphics processing unit (GPU)
6. 32GB DDR4 RAM.

The [CUDA](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#title-new-features) and [cuDNN](https://developer.nvidia.com/cudnn) were downlaoded and installed following instructions from the official [NVIDIA Website](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local).


#Procedure

1. Setting up a virtual environment

Opened Anaconda Prompt as an adminitrator and we created a virtual environment called "thesis_models" using the following command;

```
C:\> conda create -n thesis_models pip python=3.8
```

After, activate the virtual environment using,

```
C:\> activate thesis_models
```

Then, after we installed the following packages.

```
(thesis_models) C:\> pip install tensorflow==2.5.0 tensorflow-gpu==2.5.0
(thesis_models) C:\> conda install -c anaconda protobuf
(thesis_models) C:\> pip install pillow
(thesis_models) C:\> pip install lxml
(thesis_models) C:\> pip install Cython
(thesis_models) C:\> pip install contextlib2
(thesis_models) C:\> pip install jupyter
(thesis_models) C:\> pip install matplotlib
```
2. Cloning a TensorFlow object detection API repository in folder "thesis" using,

```
(thesis_models) C:\thesis> git clone https://github.com/tensorflow/models.git
```

At this stage we had something similar to:
![](https://github.com/mirugwe1/bird_detection/blob/master/photos/images.JPG)

## Data Pre-processing

### Dataset Partitioning

Using Script: data_partitioning.py

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


### Convert *.xml to *.csv

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
