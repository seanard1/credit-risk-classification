# Module 20 Report Template

## Overview of the Analysis

With records for almost 80,000 borrowers on hand, can we build a machine-learning model that can assess the risk associated with new loans based on a number of criteria about the new borrower? 

The short answer is, yes.

Given a database containing the credit history of thousands of borrowers, we can use information about whether they defaulted on their loan and have a machine-learning model determine whether a new borrower is likely to default based on the same criteria.

The available information we have to assess includes key factors for borrowing, including the size of the potential loan, the income of the applicant, the applicant's debt and the number of accounts they hold. 

Within our sample data of 77,536 borrowers, 75,036 of them did not default on their loan, so we will have to implement some machine-leaning techniques to tackle an imbalanced classification dataset -- that is to say, to train a better model, we need to give it a larger sample of borrowers who default on their loans. 

Using a logistic regression will allow the model to determine whether, given a set of variables about a loan application, the new borrower is "safe" (unlikely to default) or "high risk" (likely to default on payments). 

But after a first pass, the logistic regression model struggled with correctly identifying high-risk individuals. This is likely due to the imbalance in the initial dataset that contained 30 "safe" examples for every one that was "high risk." By using a method called random-oversampling, which randomly duplicates parts of the dataset prior to training the model, we provide the logistical regression with a more robust dataset to train on. 

As expected, it performs better.

## Results

* Logistical Regression
    * Precision: "safe": 1.00; "high risk": 0.87
    * Recall: "safe": 1.00; "high risk": 0.89
    * Accuracy: 0.94
    
As anticipated, the initial regression model performed very well on the "safe" borrowers, with precision, recall and F1 scores of 1.00, but a balanced accuracy score of 94.4% does not provide for enough confidence in the model to use in a loan-assessment scenario. Given the average loan amount is $9,800 and the average interest rate is 7.29%, the borrowee is making a potential profit of $715 off a given successful loan. With precision and recalls scores below 90% for "high risk" assessments, there is some concern that the model would reject profits at a larger scale. We need to be sure the model won't hand out more false positives in a larger sample size. 

* Random Over-Sampling Logistical Regression
    * Precision: "safe": 1.00; "high risk": 0.99
    * Recall: "safe": 0.99; "high risk": 0.99
    * Accuracy: 0.99
    
After randomly over-sampling the minority data, even to just a 2:1 ratio, we can now be more confident that the logistical regression model will perform accurately enough on a larger dataset. 


## Summary

Initially, there were concerns with the logistical regression model because of the imbalance in the initial dataset (heavily skewed to "safe" loans), but through the use of random over-sampling to give the model more examples to train on, it's clear this model for loan assessment could be implemented for new applicants.

Most of the risk comes from a model giving out false negatives -- that is, saying that a borrower is "safe" when they are actually "high risk." This category was minimized by the model from the very start, as even the initial recall score was 1.00 (18,679 correctly predicted "safe" loans against 80 false negatives). While those 80 false negatives do represent $800,000 in potential defaulted loans (average loan is $9,800), the 18,679 correct "safe" loans represent potential profit of more than $13 million (average interest rate of 7.29%). 


While there is little risk to a false positive aside from lost interest profits, at scale those lost "safe" loans that are categorized as "high risk" can add up. That is in addition to the morality of denying someone a loan when they are truly deserving. But through the use of random over-sampling to test a larger dataset, the model performed well on "high risk" assessments as well, with an F1 score of 1.00. 

This logistical regression model on borrower assessment -- tested with random over-sampling -- is ready for implementation. 