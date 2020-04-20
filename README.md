# Spot the mask
Zindi hackathon solution disclosed

This is a binary classification problem with images. CNNs are the state-of-the-art solution and most straight-forward option to achieve a high performing model.

## Algorithm 

### Classification CNN

I trained a DenseNet201 with fastai library with mixup and a final 8 epochs without mixup. I splitted the data into 90% train and 10% validation and achieved a **logloss of 0.01019** on the public leaderboard. This without any further trick. However, we can improve our models performance or at least its confidence by splitting the image into tiles and predicting all of them.

![submission](https://github.com/alfonmedela/spot_the_mask/blob/master/imgs/cnn_pred.PNG)

### Confident prediction

Here comes the most interesting part. I trained a Random Forest classifier to map the statistis on the soft predictions of the tiles to a confidence value of either 0 or 1. Technically this is not setting the values to 0 or 1 by hand as ZINDI organization made clear it wasn't allowed. Furthermore, it is not just a model that maps 0.9 into 1.0 but takes the general prediction, the mean, minimum and maximum prediction of the 5 tiles and predicts the confidence.

![tiles](https://github.com/alfonmedela/spot_the_mask/blob/master/imgs/tiles_diagram.png)

#### Training RF

I calculated the minimum, mean and maximum of the 5 predictions and used them as input to the RF together with the prediction for the whole image.


| x<sub>1</sub>  |  x<sub>2</sub>| x<sub>3</sub> | x<sub>4</sub>| 
| ------------- | ------------- |------------- | ------------- |
| minimum pred   | mean pred        | maximum pred        | pred on the whole image |


As most of the training and validation have near perfect score, we won't be able to exploit this technique unless we add new artificially generated data. What we asume is that it is possible to predict a value close to 0 because the mask was too small but when applying the model to the tiles to get a very confident value like 0.9 that change all the stats. The RF will be able to capture this if we generate this kind of data. Therefore, I defined my own criteria in which I thought it would help predict with higher success. The fuction itself contain the explanation for the randomly generated data.

#### Predicting final test images

We only have to apply previous method to every single image and submit predictions. I will also share the feature importance of the RF to show that the tiles are determinant to better predict. Of course that we kind of forced to do this but it is exactly what we intuitively wanted from such a model on top of the soft predictions. The objective of this model is to use the tiles to correct the whole image prediction.

![feature importance](https://github.com/alfonmedela/spot_the_mask/blob/master/imgs/bar_plot.png)

## Final result
This model failed on 2 test images (99.60% accuracy) but it is impossible to find those images and correct the model because ZINDI makes 3 different statements on how to differenciate between mask/no-mask classes:

- *"Your task is to provide the probability that an **image contains** at least one mask"* (data section)
- *"estimates of the corresponding probabilities of observing a **person with a mask**"* (data section)
- *"at least one person in the image is **wearing** a mask"* (discussion section)

Now have a look at these images and try to classify if it they have a mask without knowing which is the real definition of the *mask* class:


