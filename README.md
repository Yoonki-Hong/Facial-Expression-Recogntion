# Origin: https://github.com/atulapra/Emotion-detection

# Real-Time Facial Emotion Recognition

## Abstract

* Dataset: FER2013(ICML), 35887 grayscale, 48x48
* Class: angry, disgusted, fearful, happy, neutral, sad, surprised

## Environment / Dependencies

```bash
* Ubuntu 18.04
* CUDA 10.1
* cuDNN 7.6.5
* Python 3.6.9
* tensorflow 2.1.3
* numpy 1.17.4
* opencv-python 4.1.2.30
* To install the required packages, run `pip install -r requirements.txt`.
* If there is an error message with `could not load dynamic library 'libnvinfer.so.6'~`, try:
```

```bash
pip install tensorflow-cpu==2.1.0
```

## Usage

* First, clone the repository and enter the folder

```bash
git clone https://github.com/AROMI-CapstoneWorkspace/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition
```

* If you want to view the predictions without training again, run:  

```bash
cd src
python emotions.py --mode display
```

* If you want to train this model, use:  

```bash
cd src
python emotions.py --mode train
```

## Data Preparation(optional)

* The [original FER2013 dataset in Kaggle](https://www.kaggle.com/deadskull7/fer2013) is available as a single csv file. I had converted into a dataset of images in the PNG format for training/testing and provided this as the dataset in the previous section.

* In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. I have provided the code I wrote for data preprocessing in the `dataset_prepare.py` file which can be used for reference.

## Algorithm

* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.

## References

* "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
   Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,  
   X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
   M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
   Y. Bengio. arXiv 2013.

