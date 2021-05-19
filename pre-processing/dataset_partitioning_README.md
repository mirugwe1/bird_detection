This script is used to split the dataset into training, validation and testing sets.

We first split the dataset into training and testing sets in a ratio of 80:20

And then after 20% of the training data was set aside as validation set.

usage: dataset_partitioning_script.py -x -i [PATH_TO_IMAGES_FOLDER] -r [x] where;

- PATH_TO_IMAGES_FOLDER -- represents the directory path where images and the XML annotation files are stored.
Still training and testing sets are created in the same directory.
- x -- the splitting value, for example if x = 0.2, then the training set is 80% of the whole dataset and testing set 20%.