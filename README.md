# Malignant-Lymphoma-Classification

Transfer Learning using Lymphoma Images

**Brief Description:** This project involved using deep learning techniques to classify images of three types of malignant lymphoma (CLL, FL, and MCL). I trained a Convolutional Neural Network (CNN) from scratch and performed transfer learning with pre-existing models.

### Implementation (please see notebook for each task):
* Trained a CNN model from scratch for classification.
* Applied transfer learning techniques on pre-trained models, fine-tuning the last layers to adapt to our specific task.

### **`Task 1 `** 
Read the data and construct the labels. Read the images from the data path, when images of different classes are seperatedly stored in different folders named with their label names;
construct numpy arrays for both the image data and the image labels.

![image](https://github.com/travislatchman/Malignant-Lymphoma-Classification/assets/32372013/f88d2b46-accb-407c-b1b5-689ffddd6632)


### **`Task 2 `** 
Split into train-validation-test set and data visualization, and avoid that imgaes from a same person appear simutaneously in training and testing set.

![image](https://github.com/travislatchman/Malignant-Lymphoma-Classification/assets/32372013/444946a4-7bd4-4bc1-8dd3-e971344244e6)


### **`Task 3 `** 
Construct the dataset for Pytorch

For how to construct the dataset, please refer [Reference Link](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). You may have already known that we could construct the RGB dataset directly from the input path within a defined dataset class. But here please construct a grayscale one from our image and label ndarrays as we are already at this stage

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

### **`Task 4 `** 
Create the dataloader (see the reference in Task.3, please feel free to set parameters).

### **`Task 5 `** 
Construct and Train a convolutional neural network using pytorch.[Reference Link](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

You could construct your own structure of the CNN. But there are some restrictions in this task:

1. Convolution layers should be no more than 3 layers.
2. Use linear layer(s) to condese your features after the convolution layers.
3. Batch normalization is needed. Drop out is optional.
4. Use ReLU function as your activation function.

Train the network using traing and validation dataset using your own setting (optimizer, criterion, epoch number, batch size) for training.

Evaluate the network performance using test dataset.

Accuracy on the test set: 50.77%

![image](https://github.com/travislatchman/Malignant-Lymphoma-Classification/assets/32372013/60890d10-6078-4164-b466-4ed94613ede4)

### **`Task 6 `** 

Data imbalance.

As we can see, the three classes are not perfectly balanced, which might introduce bias on between them. Please modify your loss function nn.CrossEntropyLoss() to take the class inbalance into consideration, and see if the accuracy on the same test set is improved or not compared to Task 5 when only the loss function is changed.

Accuracy on the test set with the adjusted loss function: 35.38%  

![image](https://github.com/travislatchman/Malignant-Lymphoma-Classification/assets/32372013/83b21f9b-a7c7-49da-9459-412f2eef8cac)


In my case, correcting for class imbalance led to a classifier that performed worse, which is not an expected result. I assume I may have done something wrong when calculating the weights. But in training the model after accounting for imbalances, the train loss, train accuracy, validation loss, and validation accuracy does not change. Train Loss: 0.072148  Train Accuracy: 32.00%  Validation Loss: 0.093497  Validation Accuracy: 32.35%

Strictly in terms of the confusion matrices, lets start with confusion matrix before accounting for class imbalance. \\
Correct predictions: \\
Class 1: 17 \\
Class 2: 15 \\
Class 3: 1 \\
Incorrect predictions: \\
1 instance of class 1 predicted as class 2, and 3 instances as class 3 \\
6 instances of class 2 predicted as class 1 \\
14 instances of class 3 predicted as class 1, and 8 instances as class 2  


\\
after accounting for the class imbalance, the confusion matrix is 

Correct predictions: \\
Class 1: 0 \\
Class 2: 0 \\
Class 3: 23 \\
Incorrect predictions: \\
All 21 instances of class 1 predicted as class 3 \\
All 21 instances of class 2 predicted as class 3 \\

Based on the confusion matrices, the first classifier (Confusion Matrix 1) is better overall. Although it has some difficulty in distinguishing between the three classes, it has a higher number of correct predictions for class 1 and class 2.

The second classifier (Confusion Matrix 2) appears to only predict class 3, which indicates that it might be biased towards class 3 or not properly trained. The classifier does not correctly predict any instances of class 1 or class 2, making it less effective than the first classifier.

### **`Task 7 `** 
Use pre-trained Res50 Model and finetune it for 3 classes in pytorch using the Lymphoma data. Train for 10 epoches, and in the first 5 epoches, do not freeze the parameters and train the whold model; in the last 5 epoches, freeze the part of Res50 and only train your classifier (last several fc layers). To achieve this, you may need to define a new training function Train_model_new()

Please consider the class imbalance. Show performance using accuracy and confusion matrix.
