# Smart-Surveillance-System
The project "Smart Surveillance System" uses a fine-tuned version of the YOLOv5 model to detect fire and sabotage-related incidents in a Live Camera feed. 

System Requirements
• The system should process the video input from the user only if the video is of a valid format.
• The system should display the error message to the user if the video is not of a valid format.
• The system should convert the video to images.
• The system should detect the image and clarify the scenarios using classification algorithms.
• The system should generate relative output on the basis of detection.
• 64-bit, x86 desktop or laptop with dedicated gpu. The lowest compatible desktop Nvidia GPU that is supported GeForce gtx640 and laptop GPU is gtx740m and onward.
• Tensorflow with gpu support which is an open-source software library used for machine learning.
• Google object detection API
• Detectron-2 library developed by Facebook.
• Labelimg.Exe is software for annotating images and labelling objects with bounding boxes in images.

Neural network for detection of Fire Class: 
The classification of a single flame object was done after taking in account the training resu8lts of several different classifiers and neural networks was designed that had the least error rate and the highest accuracy. The
parameters of each classifier are as:
1 : Classifier A
• First hidden layer: 32 neurons.
• Second hidden layer: 32 neurons.
• Positive training examples: 1000.
• Negative training examples: 1000.
2: Classifier B:
• first hidden layer: 32 neurons.
• second hidden layer: 64 neurons.
• positive training examples: 990.
• negative training examples: 990.
3: Classifier C:
• first hidden layer: 32 neurons.
• second hidden layer: 32 neurons.
• positive training examples: 500.
• negative training examples: 500.
4: Classifier D:
• first hidden layer: 32 neurons.
• second hidden layer: 64 neurons.
• positive training examples: 500 images with plain backgrounds.
• negative training examples: 500.
With the increase in the number of neurons in each layer, complexity and computation time for testing and training the network increases but the performance of the network does not changes after a certain level. If the number of neurons used was very low, neural networks would not be able to classify images properly. Therefore, an optimum number is selected by experimentation.
