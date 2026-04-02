# Get started

## Lab session description and database

Diabetic Retinopathy (DR) is the leading cause of blindness in the working-age population of the developed world. World Health Organization estimates that 347 million people have the disease worldwide. Diabetic Retinopathy (DR) is an eye disease associated with long-standing diabetes. Around 40% to 45% of Americans with diabetes have some stage of the disease. Progression to vision impairment can be slowed or averted if DR is detected in time, however this can be difficult as the disease often shows few symptoms until it is too late to provide effective treatment.

Our goal is to develop a CNN providing an automatic diagnosis of DR with color fundus photography as input. The need for a comprehensive and automated method of DR screening has long been recognized, and previous efforts have made good progress using image classification, pattern recognition, and machine learning.

You are provided with a large set of high-resolution retina images taken under a variety of imaging conditions. Left and right fields can be provided indistinctively. Images are labeled with the following data:

1.- An image id. 2.- An eye indicator (left,right) 3.- A label, indicating the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale: 0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR

Figure 1 shows a visual example of a eye color fundus:

Image of retina

The dataset contains 3500 images divided into 3 sets:

Training set: 2000 images
Validation set: 500 images
Test set: 1000 images.
Additionally, there is a csv file for each dataset (training, validation and test) in which each lines corresponds with a clinical case, defined with two fields separated by commas:

the numerical id of the lesion: that allows to build the paths to the image.
2.- An eye indicator (0-left eye and 1-right eye). It allows mirroring right-eye images so that both left and right eyes are comparable.
the lesion label: available only for training and validation, being an integer between 0 and 4: 0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR. In the case of the test set, labels are not available (their value is -1).
For simplicity, we will transform the original labels to produce a binary label: 0 - No DR, 1 - DR. Therefore we tackle a binary classification problem.

Students will be able to use the training and validation sets to build their solutions and finally provide the scores associated with the test set.

## Design and implementation of the diagnosis system

This practice provides guidelines to build a baseline reference system. To do so, we will learn two fundamental procedures:

Process your own database with PyTorch
Design a feedforward CNN that processess images and provides a diagnostic
Use a regular network that has been pretrained using a large-scale general purpose dataset and fine-tune it for our diagnostic problem

## Evaluation Metric: AUC

We will use the area under the ROC or AUC (https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

AUC is a metric that avoids setting a specific threshold to make detections and is applied over the soft outputs of a binary classifier. By modifying the value of the threshold, we can build a ROC curve setting the False Positive Rate (FPR) in the y-axis and the True Positive Rate (TPR) in the x-axis. TPR is the proportion of positive cases than have been succesfully detected, whereas FPR is the number of false detections divided by the number of negatives.

For low detection thresholds and an imperfect system, TPR will be high at the expense of a high FPR (as the system always says 1). For high threholds, the opposite situation happens. Once the ROC is built, the AUC measures the integral behind the curve, which is in the range [0,1].

Although, at least theoretically, AUC can be lower than 0.5, in partice, the output of a baseline system that randomly decides 0 or 1 with equal probability obtains an AUC=0.5, so lower values are usually caused by bugs in the code (and could be avoided just by inverting the outputs of the system).

As we have mentioned, AUC is a metric to evaluate binary problems (labels 0,1), and has the advantage of being independent of the detection threshold. It also behaves well against unbalanced problems, as it evaluates the ranking of the scores (their order with respect to the labels) and not their absolute values.

Since our problem is binary (DR vs no DR), we will use AUC to assess the performance, with the following method from scikit-learn:

auc = metrics.roc_auc_score(labels, scores)

## Terms

1. Evaluation criteria

The evaluation of this practice will be done through a challenge, for which the students will have to send the results based on the test data in two categories:

CUSTOM Category: The results on the test set of a custom network, created from scratch by the students. In this case, the complete code of the network must appear in cells of the notebook and it will not be possible to make use of external networks/packages or pre-trained models.
FINE-TUNING Category: The results on the test set of a network that has been previously initialized in another database (with fine-tuning). In this case, existing models in torchvision or even external networks can be used.
In addition, the final mark will depend both on the results in both categories and on the content of a brief report (1 side for the description, 1 side for extra material: tables, figures and references) where they will describe the most important aspects of the proposed solutions. The objective of this report is for the teacher to assess the developments/extensions/decisions made by the students when optimizing their system. You do not need to provide an absolute level of detail about the changes made, just list them and briefly discuss the purpose of the changes.

Rules for the Codabench challenge

To participate in the challenge, the following instructions and rules must be taken into account:

Timeline: The challenge will take place between Wednesday, March 25, and Tuesday, April 21. Consequently, no solution uploads will be permitted outside of this date range.

Submission Limit: During the challenge, participants are allowed to upload four solution files per day, corresponding to different system implementations.

Code Verification: Project code will be reviewed to verify the authenticity of the results obtained during the challenge. If the results cannot be reproduced using the submitted system, the team will be disqualified from the competition and will not receive the corresponding points toward the final project grade.

Project evaluation

The evaluation of this assignment, over 10 points, is structured as follows:

4 points: Technical content of the report.

4 points: Competition results on Codabench. The score will be determined via linear regression based on the best and worst results obtained in the competition.

2 points: Quality of the submitted Python code.

2. Submission details for evaluation

Model Outputs: The .csv files containing the test outputs for the models trained in both categories must be uploaded via the Codabench platform throughout the competition (maximum of 2 submissions per day, 100 total submissions).Each file will contain a matrix of size 1000x1, with the DR score for each of the 1000 images in the test dataset. The array should be provided in text format (with 1 number per row). The reference notebook provides the code to generate these outputs.
Additionally, each group must upload a ZIP file to Aula Global containing:

Two .csv files with the test outputs of the models trained in the two categories. Each file will contain a matrix of size 1000x1, with the DR score for each of the 1000 images in the test dataset. The array should be provided in text format (with 1 number per row). Code to generate the outputs is provided later.

The report as described above.

The notebook that integrates the creation of both models so that the teacher can check how things have been implemented.

IMPORTANT: It is mandatory for each group member to focus on specific improvements or experiments. Furthermore, the report must clearly indicate the contributions of each member so that individual work can be assessed.

The project submission deadline is Tuesday, April 21, at 11:59 PM.
