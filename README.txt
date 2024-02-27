README

Name : Mayukha Thumiki
UTA ID : 1002055616
Programming language used : Python
Python Version : 3.9.13

Code Description:
The provided Python script implements a decision tree and random forest algorithm for classification. The decision tree is constructed using either an optimized or randomized approach, and the random forest is built by combining multiple decision trees.
The structure of the code is as follows:
- TreeNode Class: Represents a node in the decision tree with attributes such as feature, threshold, left and right child nodes, and a value for leaf nodes.
- load_data Function: Reads the data from a file, separating features and labels.
- Information Gain Functions: Calculate the information gain and entropy used for decision tree construction.
- Build_tree Function: Recursively builds a decision tree based on either optimized or randomized feature selection. It also includes pruning to improve efficiency.
- Predict Functions: Predicts the class label for a given input using a single decision tree or a random forest.
- Evaluation Functions: Evaluate the accuracy of the decision tree or random forest on a dataset.
- Print_results Function: Prints the results, including the predicted class, true class, and accuracy for each object.
- Main Function: Handles the main execution flow, loading data, building trees or forests, and evaluating the classification accuracy.

Instructions to Execute:
1. Install Python on the system.
2. Include the training and test datasets in the same folder as the code.
3. Open Terminal or command line prompt.
4. Navigate to the folder containing the code file and execute the following command:

		python/python3 <filename> <training_file> <test_file> <option>

For example:	python/python3 dtree.py yeast_training.txt yeast_test.txt optimized


Available options: [optimized, randomized, forest3, forest15]

Note:
The script supports both decision tree and random forest classification.
For forest options, the number in the option (e.g., 'forest3') specifies the number of trees in the random forest.
The script takes three command-line arguments: training file, test file, and option.
If the number of command-line arguments is incorrect, a usage message will be displayed.
