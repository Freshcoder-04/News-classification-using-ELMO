## 1. Introduction:

This repository contains code which uses the ELMO architecture to generate word embeddings and then uses it in a downstream task on the news-classification dataset. I have also compared the performance of ELMO embeddings with Static word embeddings for the same task on the same dataset.

## 2. ELMO

- **Embedding Layer (e0):**  
  Converts input token indices to non-contextual word embeddings (dimension = embed_size, here 100).

- **First BiLSTM (e1):**  
  Processes the embeddings to capture context; outputs concatenated forward and backward hidden states 
  $$dimension = hidden\_size * 2$$
  Here, 
  $$ dimension = 512 $$

- **Second BiLSTM (e2):**  
  Further refines context representations; again outputs a $512$-dimensional vector per token (split into forward and backward halves).

This is stored in: ``./models/bilstm.pt``

## 3. Classification and Combining methods
We then make classifiers using any type of RNN (here LSTM has been used) where the input embeddings are the embeddings of generated by the ELMO model combined using different methods.

$$ combined\_embeddings = \lambda_1*e0 + \lambda_2*e2 + \lambda_3*e3 $$

### 3.1 Frozen lambdas
We take random values of $\lambda's$ and then freeze them. Then we train a classifier model on the news-classification dataset and the model learns according to these frozen $\lambda's$.

### 3.2 Trainable lambdas
We assign values to the $\lambda's$ and then train the classifier without freezing the lambdas. This trains the $\lambda's$ as well. 

### 3.1 Learnable function
Here, instead of combining using lambdas, we learn a neural network (MLP) in order to combine the three embeddings.

## Static Embeddings
Here the pretrained models for svd, cbow and skipgram have been using from an existing repository for the same classification task. You can check it out [here](https://github.com/Freshcoder-04/Static-Word-Embeddings).


## 4. Performance Analysis Report for News Classification Models

### 4.1 Models Evaluated

1. **ELMo-based Classifiers:**
   - **ELMo Frozen:** Uses fixed (non-trainable) lambda weights to combine the three ELMo representations.
   - **ELMo Trainable:** Uses trainable scalar weights to combine the three ELMo representations.
   - **ELMo Learnable:** Uses a learnable function (MLP) to fuse the three ELMo representations.

2. **Static Embedding Classifiers:**
   - **CBOW:** Classifier built on embeddings generated via the Continuous Bag-of-Words model.
   - **SkipGram:** Classifier built on embeddings generated via the SkipGram model.
   - **SVD:** Classifier built on embeddings obtained from Singular Value Decomposition of the co-occurrence matrix.

---
### 4.2 Overall Test Set Performance
| Model             | Accuracy |  Recall | Macro F1-Score |
|-------------------|----------|--------------|----------------|
| **ELMo Learnable**    | 87.93%   | 87.93%       | 87.92%       |
| **SkipGram**          | 87.76%   | 87.76%       | 87.73%       |
| **ELMo Trainable**    | 86.47%   | 86.47%       | 86.53%       |
| **ELMo Frozen**       | 85.80%   | 85.80%       | 85.76%       |
| **CBOW**              | 84.84%   | 84.84%       | 84.78%       |
| **SVD**               | 78.93%   | 78.93%       | 78.88%       |

---
### 4.3 Detailed Results and Analysis

### 1. ELMo-based Classifiers

#### **ELMo Frozen**
- **Train Confusion Matrix:**
  ```
  [[28071   438   623   868],
   [  148 29620    91   141],
   [  460   307 26765  2468],
   [  568   376  1092 27964]]
  ```
- **Train Accuracy:** 93.68%
- **Test Confusion Matrix:**
  ```
  [[1637   69   82  112],
   [  47 1798   30   25],
   [  77   58 1508  257],
   [  89   56  177 1578]]
  ```
- **Test Accuracy:** 85.80%

**Analysis:**  
The fixed combination method works well, but it is outperformed by the methods that allow some adaptivity. Misclassifications are more frequent in classes 3 and 4 on both training and test sets.

---

#### **ELMo Trainable**
- **Train Confusion Matrix:**
  ```
  [[29373    73   141   413],
   [   62 29730    80   128],
   [  102    33 29038   827],
   [   99    33   222 29646]]
  ```
- **Train Accuracy:** 98.16%
- **Test Confusion Matrix:**
  ```
  [[1646   46   92  116],
   [  44 1766   29   61],
   [  70   28 1515  287],
   [  60   31  164 1645]]
  ```
- **Test Accuracy:** 86.47%

**Analysis:**  
Trainable lambda weights lead to very high training accuracy and a more precise confusion matrix, indicating better class discrimination. The slight improvement in test performance suggests the additional flexibility helps generalize slightly better.

---

#### **ELMo Learnable**
- **Train Confusion Matrix:**
  ```
  [[28341   327   675   657],
   [   81 29783    67    69],
   [  248   200 28174  1378],
   [  295   245  1104 28356]]
  ```
- **Train Accuracy:** 95.55%
- **Test Confusion Matrix:**
  ```
  [[1670   58   85   87],
   [  32 1821   18   29],
   [  63   29 1591  217],
   [  73   44  182 1601]]
  ```
- **Test Accuracy:** 87.93%

**Analysis:**  
Using an MLP for fusion results in the best test performance among the ELMo models. Although its training accuracy is slightly lower than the trainable version, its test performance is higher, indicating better generalization—likely due to a more effective, non-linear combination of the ELMo embeddings.

---

### 2. Static Embedding Classifiers

#### **CBOW**
- **Train Confusion Matrix:**
  ```
  [[28854   317   454   375],
   [  185 29630    76   109],
   [  664   201 27529  1606],
   [ 1024   217  1309 27450]]
  ```
- **Train Accuracy:** 94.55%
- **Test Confusion Matrix:**
  ```
  [[1659   74   95   72],
   [  56 1777   26   41],
   [ 113   54 1500  233],
   [ 135   50  203 1512]]
  ```
- **Test Accuracy:** 84.84%

**Analysis:**  
The CBOW-based classifier performs moderately well on training data, but test performance is lower compared to some of the ELMo and SkipGram variants. This may be due to CBOW’s averaging nature, which might lose some finer contextual distinctions.

---

#### **SkipGram**
- **Train Confusion Matrix:**
  ```
  [[28937   233   396   434],
   [  126 29772    28    74],
   [  372   326 27156  2146],
   [  451   243   687 28619]]
  ```
- **Train Accuracy:** 95.40%
- **Test Confusion Matrix:**
  ```
  [[1691   52   72   85],
   [  40 1816   18   26],
   [  78   41 1520  261],
   [  85   35  137 1643]]
  ```
- **Test Accuracy:** 87.76%

**Analysis:**  
The SkipGram classifier achieves high accuracy and a nearly diagonal confusion matrix on both train and test sets, indicating robust semantic representations that work well for classification.

---

#### **SVD**
- **Train Confusion Matrix:**
  ```
  [[27521   557   683  1239],
   [  468 29030   178   324],
   [  955   355 25383  3307],
   [  861   333  1478 27328]]
  ```
- **Train Accuracy:** 91.05%
- **Test Confusion Matrix:**
  ```
  [[1580  114   89  117],
   [ 118 1651   40   91],
   [ 138   68 1325  369],
   [ 118   90  249 1443]]
  ```
- **Test Accuracy:** 78.93%

**Analysis:**  
The SVD-based classifier has the lowest performance overall. The confusion matrices show more off-diagonal errors, indicating that the SVD embeddings (derived from a linear decomposition of co-occurrence counts) may not capture as much discriminative information for this task.

---

## Ranking of Models (Based on Test Accuracy)
1. **ELMo Learnable**: 87.93%
2. **SkipGram**: 87.76%
3. **ELMo Trainable**: 86.47%
4. **ELMo Frozen**: 85.80%
5. **CBOW**: 84.84%
6. **SVD**: 78.93%

---

## Hyperparameter Settings and Their Impact

- **Number of Epochs (15):**  
  Training for 15 epochs allowed the models to converge. The ELMo-based models reached very high training accuracy, and slight differences in training vs. test performance suggest good generalization for the learnable fusion method.

- **Learning Rate (0.0005):**  
  A learning rate of 0.0005 provided a stable convergence for the classifiers. This rate is low enough to fine-tune the fusion layers in ELMo-based models without causing instability.

- **RNN Architecture:**  
  All classifiers used a bidirectional LSTM with a hidden size of 128 (resulting in a 256-dimensional final state for bidirectional models). This configuration appears to be a good balance between capturing context and maintaining generalization.

- **Fusion Techniques in ELMo-based Models:**
  - *Fixed (Frozen):* Simpler but less flexible; yields decent performance.
  - *Trainable:* Slightly better fitting to training data; improves performance marginally.
  - *Learnable (MLP):* Provides non-linear fusion of the three representations, leading to the best generalization on test data.
  
- **Confusion Matrices Impact:**  
  The confusion matrices for the ELMo-based methods are more diagonal (especially for the trainable and learnable variants), indicating fewer misclassifications. The SkipGram model's confusion matrix is also highly diagonal, reflecting strong discriminative power. In contrast, SVD’s confusion matrices show more off-diagonal entries, explaining its lower accuracy.

---

## Conclusion

Among all models, the **ELMo Learnable** classifier and the **SkipGram** classifier demonstrate the best performance on the test set, with accuracies of approximately 87.9% and 87.8%, respectively. The flexible, non-linear fusion (MLP) in the ELMo Learnable model appears to offer the best trade-off between training fit and generalization, while the SkipGram embeddings provide rich semantic information that benefits classification.

**Recommendations:**  
- **For best performance:** Retrain using the ELMo Learnable method or SkipGram embeddings.
- **Hyperparameter settings:** Maintain 15 epochs, a learning rate of 0.0005, and a bidirectional LSTM with a hidden size of 128 for optimal balance between complexity and generalization.
- **Future Improvements:** Consider aligning the output layer to the actual number of classes (4 instead of 10) to further improve performance and reduce unnecessary parameters.

The detailed confusion matrices (provided above) corroborate these findings by showing clear class separation for the top-performing models.