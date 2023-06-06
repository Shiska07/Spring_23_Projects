Data files:
1. data.txt : complete data file for the project
2. 2_a_train.txt: Training data for part 2 a). This files was also used in 2 b) to test implementation.
3. 2_a_test.txt: Test data for part 2 a). This files was also used in 2 b) to test implementation.
4. 2_c_d_e_eval.txt: Evaluation data for part 2 c). d) and e)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Jupyter Notebooks with code:

1. knn.ipynb: 
Run this notebook(Cell -> Run All) and the following prompts will appear for after running code cell 6:

Enter file containing training data:
Enter file containing test data: 
Enter value of k:
Would you like to print detailed results?(y/n):

Notes: 
- The above prompt is for part 2 b) of the assignment.
- Sample format for the files is provided in notebook markdown.
- input 'y' for 'Would you like to print detailed results?(y/n): 'will print detailed results which include: i) Class labels and distance of the k nearest neighbors from the test datapoint, ii) Posterior probability of each class label for the prediction and iii) Final prediction and whether a tie had occured.
- input 'n' for 'Would you like to print detailed results?(y/n): will only print the predicted class label and the corresponding probability values.
- Tie breaker: If two or more classes have the same highest probability value, class label of the closest neighbor among them is selected.
  
----------------------------------------------------------------
The following prompts will appear for after running code cell 7:

Enter filename for leave-one-out evaluation data: 
Would you like to print detailed results?(y/n):

- The above prompt is for part 2 c) through e) of the assignment.
- Sample format for the files is provided in notebook markdown.
- input 'y' for 'Would you like to print detailed results?(y/n): 'will print detailed results which include: i) Class labels and distance of the k nearest neighbors from the test datapoint, ii) Posterior probability of each class label for the prediction and iii) Final prediction and whether a tie had occured.
- input 'n' for 'Would you like to print detailed results?(y/n): will only print the predicted class label and the corresponding probability values.
- Tie breaker: If two or more classes have the same highest probability value, class label of the closest neighbor among them is selected.
  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RESULTS notebooks:
1. raut_hw1_2c_to_2e_summarized_results.ipynb
This notebooks contains summarized results of questions 2 c) through 2 e) performed using the above mentioned datasets provided for the assignment.
Summarized results consists of the final predicted class label and posterior probabilit of the class.

2. raut_hw1_2c_to_2e_detailed_results.ipynb
This notebooks contains detailed results of questions 2 c) through 2 e) performed using the above mentioned datasets provided for the assignment.
Detailed Results consists of: 
i) Class labels and distance of the k nearest neighbors from the test datapoint. 
ii) Posterior probability of each class label for the prediction.
iii) Final prediction and whether a tie had occured.


