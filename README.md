# Xray_classification


## Dataset  

The dataset for training the models are from Kaggle ([Tuberculosis Chest X-ray Database](
https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)) and ([COVID-19 Radiolography Databse]
(https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)).

Tuberculosis (TB) Chest X-ray Database A 
team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their 
collaborators from Malaysia in collaboration with medical doctors from Hamad Medical Corporation and Bangladesh have 
created a database of chest X-ray images for Tuberculosis (TB) positive cases along with Normal images. In our 
current release, there are 700 TB images publicly accessible and 2800 TB images can be downloaded from NIAID TB 
portal[3] by signing an agreement, and 3500 normal images. Note: -The research team managed to classify TB and Normal 
Chest X-ray images with an accuracy of 98.3%. This scholarly work is published in IEEE Access. Please make sure you 
give credit to us while using the dataset, code, and trained models. Credit should go to the following: Tawsifur 
Rahman, Amith Khandakar, Muhammad A. Kadir, Khandaker R. Islam, Khandaker F. Islam, Zaid B. Mahbub, Mohamed Arselene 
Ayari, Muhammad E. H. Chowdhury. (2020) "Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, 
Segmentation and Visualization". IEEE Access, Vol. 8, pp 191586 - 191601. DOI. 10.1109/ACCESS.2020.3031384. (from 
original data source's description). 

A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their 
collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray 
images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal, 
and other lung infection dataset is released in stages. In the first release, we have released 219 COVID-19, 
1341 normal, and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, we have increased the COVID-19 
class to 1200 CXR images. In the 2nd update, we have increased the database to 3616 COVID-19 positive cases along 
with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection), and 1345 Viral Pneumonia images. We will continue 
to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients. 


## Model Building

I used `Google Colab` with GPU sessions for training the model. 

Since there are 3500 normal chest X-rays and 700 TB X-rays, 2240 normal X-rays and 448 TB X-rays for training the model, 
560 normal X-rays and 112 TB X-rays for validation, and 700 normal X-rays and 140 TB X-rays for testing. 
During training, the images are augmented by random rotation with 10 %, random translation, input image size with 
224 pixels for height, 224 pixels for width and 3 color channels. I used based transfer learning method for deep 
convolutional neural networks (CNNs). The models that I built are   
1. ResNet-50   
2. fine-tune the last 4 layers of ResNet-50    
3. ResNet-101    
4. fine-tune the last 7 layers of ResNet-101    
5. VGG-16    
6. VGG-19    
7. DenseNet-201    
8. fine-tune the last 7 layers of DenseNet-201    
9. MobileNet-V2    
10. fine-tune the last 5 layers of MobileNet-V2       

For COVID-19 data, there are 3616 COVID, 1345 Viral Pneumonia, 6012 Lung Opacity and 10192 Normal CXRs. 
The models that I built are    
1. ResNet-50   
2. VGG-19   
3. ResNet-101    

In this web application, I used ResNet50 for TB and VGG-19 for COVID classification. 

I used the pre-built weights from `ImageNet`. After these layers, the following layers were built. 
- GlobalAveragePooling2D
- Dropout layer with 0.25 probability
- Dense layer with 128 neurons with activation Rectified Linear Units (ReLU)
- Dropout layer with 0.25 probability
- Dense layer with 1 neurone with sigmoid activation as last layer
Compilation steps are followed. 
- Adam optimizer with learning rate 0.001
- binary cross-entropy for loss        
The results can be seen at the next page.      

## References    
- [Reliable Tuberculosis Detection Using Chest X-ray With Deep Learning, Segmentation and Visualization](https://ieeexplore.ieee.org/document/9224622)  
- [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning?utm_source=gg&utm_medium=sem&utm_campaign=17-DeepLearning-ROW&utm_content=B2C&campaignid=6465471773&adgroupid=76541824319&device=c&keyword=certificate%20in%20deep%20learning&matchtype=b&network=g&devicemodel=&adpostion=&creativeid=506851063340&hide_mobile_promo&gclid=CjwKCAiA3L6PBhBvEiwAINlJ9NMH7eBJHkcMxL07a9G52Jbp9MT8HlYoGUxywpwlFxOWi8tgg3vKARoCG9oQAvD_BwE)
- [Deep Learning with Python by Francosis Chollet](https://www.manning.com/books/deep-learning-with-python-second-edition)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurelien Geron](https://www.manning.com/books/deep-learning-with-python-second-edition)
