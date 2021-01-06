# Image Recognition of Handwritten Digits
**Keywords** - Computer Vision, Deep Learning, Python, Keras

### Quick Project Description -
  - Achieved 99.1% accuracy on 10-class classification problem of MNIST dataset containing 60,000 images of handwritten digits. 
  - Built, trained and tuned **Convolutional Neural Network** with Adam optimizer, ReLU activation function and image augmentation for the purpose of addressing the problem of overfitting


### Project Repo Navigation
  - `Development` folder contains python code files to train a CNN deep learning model
     - `mnist_cnn.py` trains the CNN model without image augmentation
     - `mnist_cnn2_augmentation.py` trains the CNN model using image augmentation
     - `mnist_image_dataset_creation.py` pre-processes the raw data, and also segregates images of each digit into separate folders. This organisation can be seen in the `mnist_dataset` folder.
  - `mnist_dataset` folder contains the pre-processed dataset segregated into train & test folders
  - At `Root` location
    - Project Presentation (`Project_Presentation.pptx`)
    - Final trained CNN model (`mnist_model_2.h5`)


