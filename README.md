# BayesNAS
code for paper *BayesNAS: A Bayesian Approach for Neural Architecture Search*

## Description
The algorithm is a one-shot based Bayesian approach for neural architecutre search. It is capable of finding efficient network architecture for image classification task. This approach is also very flexible in the sense of finding good trade-off between classification accuracy and model complexity.

## RNN
Besides convolutional neural networks, we also provide theoretical analysis about sparse prior and Hessian computation for recurrent layers in [RNN](./RNN.pdf).

## Search
To run the searching algorithm, please go to [search](./search) and run *main.py* with
```bash
python main.py --lambda_child your_lambda_child --lambda_origin your_lambda_origin
```
A folder is expected to appear in the same directory containing all parameters of our algorithm wrt. to the normal cell and reduct cell.  

## Cell Evaluation
In the [search](./search), please load the corresponding *gamma_norm.pkl* and *gamma_reduct.pkl* for cell selection and building. Then replace the *NormalCell* and *ReductCell* in *cell.py* in [CNN](./CNN) manually.

(Unfortunately we don't support automation of this process at the moment.)

## Pre-trained
we also provide our best pre-trained model in [pre_trained](./CNN/pre_trained) for different lambdas. You can find them according to the corresponding lambda.

