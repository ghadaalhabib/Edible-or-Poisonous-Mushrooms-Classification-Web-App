# Edible-or-Poisonous-Mushrooms-Classification-Web-App
This is a Python code for a web application that classifies mushrooms as edible or poisonous using machine learning models.


<img width="1000" alt="Screen Shot 2023-05-01 at 2 36 33 AM" src="https://user-images.githubusercontent.com/67872328/235381292-b0d9de4c-1932-4477-bae2-b53e2678d4e2.png">

## Classifier Selection: 

<img width="550" alt="Screen Shot 2023-05-01 at 2 42 14 AM" src="https://user-images.githubusercontent.com/67872328/235381495-ec2fb904-1192-4be3-a0ec-3d9e8d65d144.png">

The classifier options are shown in the sidebar, and when a classifier is selected, the hyperparameters of the model can be chosen.

### 1- Support Vector Machine(SVM): 

With the following hyperparamters: C: Regularization parameter, Kernel, and Gamma.


### 2-  Logistic Regression: 

With the following hyperparamters: C: Regularization parameter and Maximum number of iterations.

### 3- Random Forest:

With the following hyperparamters: number of trees in the forest, maximum depth of the tree, and Bootstrap samples when building trees.



## Evaluation Metrics: 
When the "Classify" button is clicked, the selected classifier's model is trained on the training set and tested on the testing set. The model's accuracy, precision, and recall scores are displayed, and it creates different plots such as confusion matrix, ROC curve, and Precision-Recall curve, based on the selected metrics from the sidebar.

<img width="550" alt="Screen Shot 2023-05-01 at 2 39 26 AM" src="https://user-images.githubusercontent.com/67872328/235381390-af2bea7d-1d9a-4955-9e65-7bdf3e6f53bf.png">
<img width="550" alt="Screen Shot 2023-05-01 at 2 39 22 AM" src="https://user-images.githubusercontent.com/67872328/235381393-56466fa0-e154-4603-a16e-2055e37c059a.png">
<img width="550" alt="Screen Shot 2023-05-01 at 2 39 14 AM" src="https://user-images.githubusercontent.com/67872328/235381394-5a70a278-d607-454e-bb89-e6214d258fbb.png">
