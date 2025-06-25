# MHGNIMC: Multi-view hybrid graph representation learning with neural inductive matrix completion for miRNA-disease association identification
Authors: Qian Weng, Yuping Sun* and Jie Ling.
*Corresponding author: syp@gdut.cn(Y. Sun)

## Introduction
In recent years, a large number of biological experiments have strongly shown that miRNAs play an important role in the development of human complex diseases. As the biological experiments are time-consuming and labor-intensive, developing an accurate computational prediction method has become indispensable to identify disease-related miRNAs. Therefore, we propose a novel method based on multi-view hybrid graph representation learning and neural inductive matrix completion for miRNA-disease associations identification, named MHGNIMC. Firstly, the multiple similarity networks for miRNAs and diseases are constructed to characterize the relationships of miRNAs and diseases. Secondly, the hybrid graph representation learning framework is introduced to learn the feature representations of miRNAs and diseases simultaneously. Finally, feature representations are input into a novel neural inductive matrix completion model to generate an association matrix completion. To verify the effectiveness of the method, we conduct a series of experiments on the Human MicroRNA Disease Database v3.2 (HMDD v3.2). Experimental results have demonstrated the excellent performance of MHGNIMC in comparison with several existing state-of-the-art methods. Besides, case studies conducted on several human diseases further confirm the prediction capability of MHGNIMC for predicting potential disease-related miRNAs.

## Requirement
To run the code, you need the following dependencies:
* Python == 3.8
* pytorch == 1.12.0
* PyTorch Geometry == 2.3.1
* scikit-learn == 1.2.1
