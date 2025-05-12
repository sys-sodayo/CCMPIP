# CCMPIP

## Introduction
Proinflammatory peptides (PIPs) are short bioactive sequences that mediate immune responses and contribute to various inflammatory diseases. Accurate identification of PIPs is essential for elucidating disease mechanisms and accelerating therapeutic development. However, sequence diversity and complexity make traditional wet‐lab assays time‐consuming and costly, highlighting the need for efficient computational solutions. Inspired by the success of pre-trained protein language models (PLMs) in protein recognition tasks, we present CCMPIP, a unified framework that fuses semantic embeddings from ProtT5 with physicochemical descriptors from AAindex via a cross‐attention mechanism. Peptide sequences are first encoded into dual feature matrices, which are then integrated by cross‐attention to capture interdependencies. The resulting representation is refined through cascading convolutional neural network (CNN) layers and a capsule network to model local patterns and hierarchical features, and finally classified by multilayer perceptron (MLP) under 5-fold cross-validation. Comparative experiments against recent ensemble predictors demonstrate CCMPIP’s superior predictive power. Moreover, interpretability analyses using attention heatmaps and STREME motif enrichment confirm that CCMPIP highlights biologically relevant residues, providing transparent insights into proinflammatory activity.

![image](https://github.com/user-attachments/assets/05e51491-e678-4559-80f1-d4decf0bdc64)

- ProtT5：https://github.com/agemagician/ProtTrans

## Requirements
pytorch==2.6.0+cu126  
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
