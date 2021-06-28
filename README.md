# Dice-Recognizer

My first experience with machine learning. I figured a good starting project would be a system that could classify different types of die.

---

It's kind of a mess right now, as I'm learning and testing different models

## Implementation
After researching different ways to perform machine learning, I decided to use TensorFlow with Python.

### Steps Involved
1. Install required modules for TensorFlow
2. Find a dataset that contains many images of different kind of dice (d4, d6, d8, d10, d12, d20)
3. Learn and test many different techniques for building machine learning algorithms, specifically related to computer vision

### Current Progress
* I found a [dataset on Kaggle](https://www.kaggle.com/ucffool/dice-d4-d6-d8-d10-d12-d20-images) that I am currently using to train and test the models
* After learning how to create machine learning models with TensorFlow and Keras, I have designed a convolutional neural network that succesfully can classify different kinds of dice
* However, the model I designed currently overfits, and doesn't generalize to new data

### Future Steps
* Fix the model so that it doesn't overfit
* Refactor code so that it can be easily exported and used in other projects
