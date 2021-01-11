## Summary

> Your model should be able to learn from the data and be able to predict the objective value.

- The questions you should ask:
  - What exactly is the business objective or the general objective? (building model probably not the end goal).
  - How does the company expect to use and benefit from this model? (it will determine how you frame the problem, what algorithms you will select, what performance measure you will use to evaluate your model, and how much effort you should spend tweaking it).
  - What the current solution looks like (if there's any)? (it often give you a reference performance, as well as insights on how to solve the problem).

- A squence of data processing components is called a data _pipeline_.
- _Pipelines_ are very common in machine learning systems because there are a lot of data to manipulate and many data transformations to apply.
- Pipeline components typically run asynchronously. Each component pulls a large amount of data, process it, and spit out the result in another data store, and then the next component in the pipeline pull this data and spit out its own output, and so on.
- Each pipeline component is fairly self-contained: the interface between components is simply the data store. (Different teams can focus on different components).
- If one of the pipeline component breaks down, the downstream components can often continue to run normally for a while by just using the last output from the broken component. On the other hand, a broken component can go unnoticed if for some time if proper monitoring is not implemented.

- A typical performance measure for regression problems is _Root Mean Square Error_ (RMSE). RMSE measures the standard deviation of the errors the system makes in its predictions. <br>
For example: <br>
RMSE equal to 50.000 means that about 68% of the system's predictions fall within $50,000 of the actual value, and about predictions fall within $100,000 of the actual value. <br>
- When a feature has a bell-shaped _normal distribution_ (also called _Gaussian distribution_), the "68-95-99.7" rule applies: 68% of the values fall within standard deviation of the mean, 95% within 2 times standard deviation of the mean, and 99.7% within 3 times standard deviation of the mean.

- Even though RMSE is generally the preferred performance measure for regression task, in some context you may prefer to use another function. <br>
For example: <br>
Suppose that there are many outliers district, in that case you may consider using _Mean Absolute Error_ (MAE).

- Both RMSE and MAE are ways to __measure the distance__ between two vector: vector of predictions and vector of target values.

- List and verify the assumptions that were made so far (by you or others). <br>
For example: <br>
The district prices that your system outputs are going to be fed into a downstream Machine learning system and we assume that these prices are going to be used as such. <br>
But, what if the downstream system actually converts the prices into categories (e.g., "cheap", "medium", "expensive") and then using those categories instead of the prices themselves? In that case, getting the price perfectly right is not important at all. Your system just need to get the category right. If that the case, then the problem should have been framed as classification task and not regression task.

- The correlation coefficient ranges from -1 to 1. Here's correlation between median house value with each attributes:

Attributes | Correlation
--- | ---
median_house_value |    1.000000
median_income |         0.687160
total_rooms |           0.135097
housing_median_age |    0.114110
households |            0.064506
total_bedrooms |        0.047689
population |           -0.026920
longitude |            -0.047432
latitude |             -0.142724

  - When it is close to 1, it means that there is a strong positive correlation, for example: <br>
    The median house value tends to go up when the median income goes up. <br>
  - When it is close to -1, it means that there is a strong negative correlation, for example: <br>
    The median house value have a slight tendency to go down when you go north (from the latitude).

- The correlation coefficient only measures linear correlation (e.g., if x goes up then y generally goes up or down) so it may completely miss out on non-linear relationship (e.g., if x is close to zero then y generally goes up).
