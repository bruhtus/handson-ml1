## Summary
> Make reasonable assumption about the data and you evaluate only a few reasonable models. <br>
> If you make absolutely no assumption about the data then there's no reason to prefer one model over any other (No Free Lunch theorem). <br>

- Machine learning is a program that learns from data.
- Machine learning help humans learn, not replace them.
- Machine learning can reveal unsuspected correlations or new trends, and thereby lead to better understanding of the problem.
- Applying machine learning to dig into large amounts of data can help discover patterns that were not immediately apparent.

- Supervised learning:
  - the training data you feed to the algorithm includes the desired solutions called _labels_.
  - A typical tasks are classification and to predict a target.
  - A few example of supervised learning: KNN, Linear Regression, Logistic Regression, SVM, Decision Tree, Random Forests, Neural Networks.

- Unsupervised learning:
  - The training data is unlabeled.
  - Tries to learn without a teacher.
  - A few example of supervised learning:
    - Clustering: K-means, Hierarchical Cluster Analysis, Expectation Maximization.
    - Visualization and dimensionality reduction: Principal Component Analysis, Kernel PCA, Locally-Linear Embedding (LLE), t-distributed Stochastic Neighbor Embedding (t-SNE).
    - Association rule learning: Apriori, Eclat.

- Run clustering algorithm to try to detect groups of similar attributes.
- Dimensionality algorithm is to simplify the data without losing too much information.
- Reduce the dimension of your training data using a dimensionality reduction algorithm before you feed it to another machine learning algorithm to make it run faster, take up less space, and in some cases it may perform better.
- Most semi-supervised learning algorithms are combinations of unsupervised and supervised algorithms.

- Reinforcement learning can observe the environment, select and perform actions, and get rewards or penalties in return.
- Reinforcement learning learn by itself what is the best strategy (called policy) to get the most reward over time.
- A policy defines what action the agent (learning system) should choose when it is in a given situation.

- Batch learning: must be trained using all the available data. To know about new data, you need to train the new version of the system from scratch (not just new data but also old data).
- Online learning (incremental learning): feed the data individually or by small groups called mini-batches. Once the system has learned about new data, it doesn't need them anymore.
- Learning rate: how fast the system should adapt to changing data.

- High learning rate will rapidly adapt the system to new data but it will also quickly forget the old data.
- Low learning rate will learn more slowly but also be less sensitive to noise in the new data.

- Instance-based learning system learns the examples then generalizes to new case using a similarity measure.
- Model-based learning system build a model of the examples then use that model to make predictions.

- __How can you know which value will make your model perform best?__ You need to specify a performance measure.

- You can define a __utility function__ to measure how good the model is, or __cost function__ that measures how bad it is.
- Two things that can go wrong: __bad algorithm__ and __bad data__.
- We may want to consider the trade-off between spending time and money on algorithm development versus spending it on corpus development.
- Training data must be representative of the new cases you want to generalize to.

- Take some time to cleaning up your training data.
- If some intances are cleary outliers, it may help to simply discard them or try to fix the errors manually.
- If some intances are missing a few features, you must decide whether you want to ignore that attributes, fill in the missing values, or train one model with feature and one model without it.
- Overfitting: the model performs well on the training data but doesn't generalize well.
- __The model is likely to detect patterns in the noise itself__.

- Balance between fitting the data perfectly and keeping the model simple enough to ensure that it will generalize well.
- Underfitting: model is too simple to learn the underlying structure of the data.
- Split the data: training dataset and testing dataset, if you need it, also use validation dataset (split from training dataset).
