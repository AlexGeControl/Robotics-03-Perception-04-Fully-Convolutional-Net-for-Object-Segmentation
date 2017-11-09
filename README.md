## README

### Technical Report for Fully-Convolutional Network based Follow Me

---

<img src="writeup_images/demo.gif" width="100%" alt="Fully-Convolutional Network for Object Tracking" />

The goals of this project are the following:
* Use the simulator to collect data of to-be-tracked person
* Build, a fully-convolutional network in Keras that segments the to-be-tracked person
* Train and validate the model with a training and validation set
* Test that the model successfully tracks the target person

---

### Files Submitted

---

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **code/model_training.ipynb**: Jupyter notebook used to train the FCN.
* **docs/report/model_training.html**: HTML version of the Jupyter notebook.
* **README.md**: writeup report for technical review
* **data/weights/config_model_weights & data/weights/model_weights**: deployed FCN model for simulator control

After downloading the deployed model, it can be used to control the quad-rotor through the following commands:

```shell
python follower.py model_weights
```

---

### Model Architecture and Training Strategy

---

#### 1. Network architecture

#### 2. Hyperparameter tuning

#### 3. Comparison between fully-convolutional & fully-connected layers

#### 4. Data manipulation

#### 5. Generalization of current network

**The network cannot be used to track either other objects like dog, cat, car or other specific person.** The reasons are as follows:

* The network is trained in a supervised way. However, only 3 types of labeled data, namely target person, other person and background, are provided.

Since no data is provided for definition of other objects, the network cannot be used to track them.

**In order to use the network for tracking of other objects**, following modifications can be applied to the training procedure

1. Collect images containing target objects with pixelwise labeling.
2. If the collected dataset is large enough, train a network from scratch.
3. If the collected dataset is only of limited size, transfer a pre-trained network attained on a huge & diverse dataset like VGG16 and fine-tune it on collected data, as it did in the original FCN paper.

---

### Model Performance

---

#### The neural network should obtain an accuracy greater than or equal to 40% (0.40) using the Intersection over Union (IoU) metric

According to the Jupyter notebook attained after an arduous training, the metrics and corresponding values are as follows:

|    Metric   |  Value |
|:-----------:|:------:|
|   Accuracy  | 0.7588 |
|     IOU     | 0.5452 |
| Final Score | 0.4137 |

They meet the performance requirement.
