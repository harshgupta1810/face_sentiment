# Facial Expression Recognition using Deep Learning

This project is designed to recognize facial expressions using deep learning models. The code in this project utilizes the Facial Expression Recognition Challenge dataset, which contains 35,000 images categorized into 8 classes representing different facial expressions.

## Table of Contents

1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Preprocessing](#preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Sample Predictions](#sample-predictions)
9. [Save Model](#save-model)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgments](#acknowledgments)
13. [Documentation](#documentation)

## Project Description

This project aims to build a deep learning model for facial expression recognition. The model is trained on the Facial Expression Recognition Challenge dataset, which contains 35,000 facial images categorized into 8 classes representing different emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Dataset

The model is trained on the Facial Expression Recognition Challenge dataset, which contains 35,000 facial images in 8 classes representing different facial expressions.

For more details about the dataset, you can visit the following link: [Facial Expression Recognition Challenge Dataset](https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge)

## Installation

To run this project, you need to install the required dependencies. The code uses Python and various libraries, including TensorFlow, Keras, Pandas, NumPy, Matplotlib, and Seaborn. You can install the required dependencies using the following commands:

```bash
!pip install tensorflow
!pip install keras
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
```

## Preprocessing

The code includes preprocessing steps to convert the image pixels into arrays and one-hot encode the emotion labels. The data is split into training and validation sets to train and evaluate the model.

## Model Architecture

The model architecture consists of several convolutional layers with activation functions such as ELU and BatchNormalization. The model also includes Dropout layers to prevent overfitting. The final layer uses a softmax activation function to predict the facial expression class.

## Training

The model is trained using the Adam optimizer and categorical cross-entropy loss function. EarlyStopping and ReduceLROnPlateau callbacks are used to improve training efficiency and prevent overfitting. The training progress is monitored using accuracy and loss metrics.

## Evaluation

The model's performance is evaluated using accuracy and loss metrics on the validation set. The training and validation accuracy and loss are plotted for each epoch to visualize the model's learning progress.

## Sample Predictions

The trained model is used to make predictions on sample images from the validation set. The true and predicted emotions are displayed along with the corresponding sample images.

## Save Model

The trained model is saved as "facial_emotions_model.h5" for future use.

## Contributing

Contributions to this project are welcome. If you have any suggestions, bug fixes, or improvements, please feel free to open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to the creators of the Facial Expression Recognition Challenge dataset for providing a valuable resource for research and development.

## Documentation

For more details on how to use the code and the functionalities of each part, please refer to the code comments and documentation within the code files. If you have any questions or need further assistance, you can contact the project creator, Harsh Gupta (Desparete Enuf).
