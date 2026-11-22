# Cat vs Dog Classifier – Viva Questions and Answers

**1. What is image classification?**  
Image classification is a computer-vision task in which a model assigns an input image to a predefined class.

**2. What is a CNN?**  
A Convolutional Neural Network is a deep-learning model designed for grid-like data such as images. It learns visual features using convolutional filters.

**3. What is a convolution operation?**  
A small matrix called a kernel or filter slides over an input image or feature map and computes weighted sums to detect patterns.

**4. What is pooling?**  
Pooling reduces the spatial dimensions of feature maps. Max-pooling keeps the largest value from each local region.

**5. Why is image resizing required?**  
A model expects a fixed input shape. Resizing makes all images the same dimensions.

**6. What is data augmentation?**  
It creates varied training examples by applying transformations such as flipping, rotation and zooming.

**7. What is transfer learning?**  
Transfer learning reuses knowledge learned by a model on a large dataset for a new, related task.

**8. What is MobileNetV2?**  
MobileNetV2 is an efficient CNN architecture designed for image tasks, including use on devices with limited resources.

**9. What is feature extraction?**  
Feature extraction uses a pre-trained network as a fixed feature extractor while training a new classifier on top.

**10. What is fine-tuning?**  
Fine-tuning unfreezes some pre-trained layers and trains them with a small learning rate on the new dataset.

**11. Why are some layers frozen initially?**  
The early convolutional layers already contain useful general visual features. Freezing them reduces training cost and helps avoid destroying useful learned weights.

**12. What is an epoch?**  
One epoch is one complete pass through the training dataset.

**13. What is batch size?**  
Batch size is the number of training examples processed before the model updates its weights.

**14. What is binary cross-entropy?**  
It is a loss function commonly used for binary classification. It measures the difference between predicted probabilities and true labels.

**15. What is a sigmoid function?**  
The sigmoid function maps a value to a number between 0 and 1, which can be interpreted as the model's probability score for the positive class.

**16. What is overfitting?**  
Overfitting occurs when a model performs well on training data but poorly on unseen data.

**17. Why do we use validation data?**  
Validation data helps monitor generalization during training and supports model-selection decisions without using the final test set.

**18. Why do we use test data?**  
Test data gives a final evaluation on examples not used for training or model selection.

**19. What does model confidence mean in this application?**  
It is the model's predicted score for the selected class. It is not a guarantee that the prediction is correct.

**20. What is Streamlit?**  
Streamlit is a Python framework for building simple interactive data and machine-learning web applications.

**21. What is the output of this project?**  
The application predicts either Cat or Dog and displays the model's confidence score.

**22. What are the limitations of this project?**  
It only recognizes cats and dogs and may make mistakes on low-quality or unusual images.
