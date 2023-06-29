# Malignant-Lymphoma-Classification

Transfer Learning using Lymphoma Images

**Brief Description:** This project involved using deep learning techniques to classify images of three types of malignant lymphoma (CLL, FL, and MCL). I trained a Convolutional Neural Network (CNN) from scratch and performed transfer learning with pre-existing models.

### Implementation (please see notebook for each task):
* Trained a CNN model from scratch for classification.
* Applied transfer learning techniques on pre-trained models, fine-tuning the last layers to adapt to our specific task.

### **`Task 1 `** 
Read the data and construct the labels. Read the images from the data path, when images of different classes are seperatedly stored in different folders named with their label names;
construct numpy arrays for both the image data and the image labels.

Example
![image](https://github.com/travislatchman/Malignant-Lymphoma-Classification/assets/32372013/f88d2b46-accb-407c-b1b5-689ffddd6632)

Label: CLL
![image](https://github.com/travislatchman/Malignant-Lymphoma-Classification/assets/32372013/ec5b56ca-1724-4b95-add8-8bfbb3cf999a)


### **`Task 2 `** 
Split into train-validation-test set and data visualization, and avoid that imgaes from a same person appear simutaneously in training and testing set.

![image](https://github.com/travislatchman/Malignant-Lymphoma-Classification/assets/32372013/444946a4-7bd4-4bc1-8dd3-e971344244e6)


### **`Task 3 `** 
Construct the dataset for Pytorch

For how to construct the dataset, please refer [Reference Link](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). You may have already known that we could construct the RGB dataset directly from the input path within a defined dataset class. But here please construct a grayscale one from our image and label ndarrays as we are already at this stage

When defining the public values: Images, Labels are the input images and labels for the dataset; Transforms contains the transform parameters

    """
When defining the public values: Images, Labels are the input images and labels for the dataset; Transforms contains the transform parameters
Inputs:
    Images: input images in one set, numpy arrays
    Labels: input labels in one set, numpy arrays
    Transforms: containing the parameters for data transform


    *** Please convert the label from index to one-hot:
                    index      one-hot
              Label CLL: 1       [1,0,0]
              Label FL: 2       [0,1,0]
              Label MCL: 3       [0,0,1]

  """
