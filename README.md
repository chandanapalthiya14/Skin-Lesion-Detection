Skin Lesion Classification using Deep Learning

Project Description

This project implements a deep learning-based computer vision system to automatically classify skin lesions from dermoscopic images. It aims to assist in early detection of skin cancer (e.g., melanoma) using Convolutional Neural Networks (CNNs) and transfer learning.

Features

Classifies skin lesions into multiple categories (benign vs malignant or multiple types).

Uses transfer learning with architectures like EfficientNet, ResNet50, or VGG16.

Implements data preprocessing and augmentation for better generalization.

Provides Grad-CAM visualization to highlight areas influencing predictions.

Optionally deployable as a Flask web application.

Dataset

HAM10000 / ISIC 2018/2020 datasets

Contains labeled dermoscopic images of skin lesions.

Preprocessing includes resizing, normalization, and augmentation.

### Models:

The CNN model consists of four convolutional layers with max-pooling, followed by two fully connected layers with BatchNormalization and dropout regularization. The Xception model is a pre-trained deep learning model that has been trained on ImageNet and includes 126 layers. Transfer learning using Xception network was employed as a feature extractor for the categorization of skin lesions.  The rational to this is that xception which in the family of inception networks from google labs has following merit. 

1. Reduced model size and computational cost 

2. Gain in performance on CNN than other inception models

3. Trains more quickly than the VGG family

4. Accepts a lesser image dimension of (71, 71) 

#### Several adjustments were made to further optimize the model. These changes includes:

1. To learn new features from the data set, the xception model layers were left unfrozen during the training process in Google Colab.

2. Swapping out the top layers of the xception model with a Relu activated block consisting of a dropout layer (0.5), a fully connected dense layer (128) with a batch    normalization layer. 

3. In order to balance out the class disparity, a class-weighted learning technique was adopted, which involved assigning different weights to different classes in the loss function. The weights were calculated using the class_weight function from the sklearn.utils libraries. 

4. To avoid over fitting call_back functions like the EarlyStopping, ReduceLRONPlateau and ModelCheckpoint were used during the training process.
Focal loss function was applied instead of the categorical cross_entropy to improve the model performance even further.

5. The problem of class imbalance was naturally resolved by focal loss. In real-world applications, we employ a α- balanced variant of the focal loss Ɣ that combines the traits of the focusing and weighing parameters, producing accuracy that is marginally higher than the non-balanced crossentropy loss

6. To further enhance the performance of the model, a soft-attention network layer with a dropout (0.2) was added to the architecture, this helps to enhance the value of critical features while suppressing the noise-inducing features. 


#### With these ajustments we were able to create eight alternative methods: 

        CNN alone (CNN1), 

        CNN + Dropout regularization (CNN2), 

        CNN + Dropout regularization + Augmentation (CNN3), 

        CNN + Dropout + Augmentation + class_weights (CNN4), 

        CNN + Dropout + Augmentation + class_weights + soft attention layer (CNN5), 

        Xception + Dropout + Augmentation + class_weights (Xception1), 

        Xception + Dropout + Augmentation + class_weights + soft attention (Xception2), 

        Xception + Dropout + Augmentation + class_weights + soft attention + focal loss (Xception3)

Technologies Used

Python

TensorFlow / Keras

OpenCV, NumPy, Pandas

Matplotlib / Seaborn

Google Colab (optional for free GPU)

Flask (optional for deployment)

Results:

After training and evaluating both models, the Xception model (Xception3) outperformed all the other models, achieving a higher accuracy of 85.7% average f1-score of  0.72 This indicates that the pre-trained Xception model, with its advanced architecture, numerous layers, focal loss, dropouts, augmentation and class-weighted is better suited for image classification tasks such as this one.

Future Work

Multi-class classification for all lesion types.

Integration with additional datasets for broader detection.

Deploy as a web or mobile application for clinical use.
