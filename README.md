# Malignant Birthmark Recognition 
## Backend

Kotlin Backend to serve user requests for [MalignantBirthmarkRecognition_Frontend](https://github.com/wsdt/malignantbirthmarkrecognition_frontend).

### Training
The Backend uses Transfer Learning and the pre-trained model VGG with 16 layers from Oxford University. 
We provided a small shell script to make enlarge the java virtual machine boundaries to avoid OutOfMemoryExceptions. 

### Prediction
The backend provides only a single http-post-route to upload a image. This image is then classified by the 
convolutional neural network and will return a json-array with all confidence-rates. 

### Attribution
This project was inspired and is based on the [Cat and Dog Recognizer by Klevis](https://github.com/klevis/CatAndDogRecognizer). 

### Open-Source
This application is open source. Please see the current license. 

#### Pull-Requests
... are always welcome! 