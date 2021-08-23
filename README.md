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


# Procedure

## Preparing the environment

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

3. Installing Protobuf and Object Detection API

Protobufs are used  by the Tensorflow Object Detection API to configure the model and training the parameters.

This was achieved using the following command.

```
(thesis_models) C:\thesis\mobilenet\models\research> protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto

```
By running the command above, name_pb2.py file of every .proto file is created in the protos folder as seen in the image below.
![](https://github.com/mirugwe1/bird_detection/blob/master/photos/protos.JPG)

The Object Detection API is installed using object_detection package which was achived using the two commands below.

```
(thesis_models) C:\thesis\mobilenet\models\research> python setup.py build
(thesis_models) C:\thesis\mobilenet\models\research> python setup.py install
```

4. Verifying our Installation

To verify our installation, used the following command.

```
(thesis_models) C:\thesis\mobilenet\models\research> python object_detection/builders/model_builder_tf2_test.py
```

The following output was obtained, and therefore our installation was confirmed successful.
![](https://github.com/mirugwe1/bird_detection/blob/master/photos/image1.JPG)

After successfully installing the object detection API, we started on training our models.

## Training

5. Data Pre-processing

5.1 Dataset Partitioning

Using data_partitioning.py script in the pre-processing folder, we split the dataset into training, validation and testing sets.

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


5.2  Convert *.xml to *.csv

Script: xml_to_csv.py
 
The script helps in generating csv files from the XML files. 

Usage: 
Place training, validation, and testing datasets folders into a single folder named "image" but the folder's name can be changed if at all you change lines 30 and 32 in the script to change name of that folder.

Then run
```
python xml_to_csv.py install
```

5.3 Creating Label Map

A label map is required by the tensorflow in both training and detection processes. And since our our dataset has only on class "bird", we created the label map below.

```
item {
  id: 1
  name: !'bird'`#f03c15`
}

```

5.4 Convert *.xml to *.record

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

After sections 5.1 to 5.3, we copied the training set, validation set to the images folder found in C:\thesis\sdd_mobilenet\models\research\object_detection\images and the TFrecods files were copied to the object_detection folder C:\thesis\sdd_mobilenet\models\research\object_detection. Finally, we had;
 ![](https://github.com/mirugwe1/bird_detection/blob/master/photos/training.JPG)

Credit: This site was so helpful in process of installing and preparing the virtual environment: [TensorFlow-Object-Detection-API-Tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)
