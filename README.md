# Edible-or-Poisonous-Mushrooms-Classification-Web-App
This is a Python code for a web application that classifies mushrooms as edible or poisonous using machine learning models.


![Screen Shot 2023-05-01 at 2 36 33 AM](https://user-images.githubusercontent.com/67872328/235381292-b0d9de4c-1932-4477-bae2-b53e2678d4e2.png)

## Choosing the Classifier: 
<img width="550" alt="Screen Shot 2023-05-01 at 2 36 33 AM" src="https://user-images.githubusercontent.com/67872328/235381292-b0d9de4c-1932-4477-bae2-b53e2678d4e2.png">

The classifier options are shown in the sidebar, and when a classifier is selected, the hyperparameters of the model can be chosen.

### 1- Support Vector Machine(SVM): 

With the following hyperparamters: C: Regularization parameter, Kernel, and Gamma.


### 2-  Logistic Regression: 

With the following hyperparamters: C: Regularization parameter and Maximum number of iterations.

### 3- Random Forest:

With the following hyperparamters: number of trees in the forest, maximum depth of the tree, and Bootstrap samples when building trees.



## Evaluation Metrics: 
When the "Classify" button is clicked, the selected classifier's model is trained on the training set and tested on the testing set. The model's accuracy, precision, and recall scores are displayed, and it creates different plots such as confusion matrix, ROC curve, and Precision-Recall curve, based on the selected metrics from the sidebar.
