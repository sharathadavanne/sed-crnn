# Single and multichannel sound event detection using convolutional recurrent neural network
[Sound event detection (SED)](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-detection) is the task of recognizing the sound events and their respective temporal start and end time in a recording. Sound events in real life do not always occur in isolation, but tend to considerably overlap with each other.
Recognizing such overlapping sound events is referred as polyphonic SED. Performing polyphonic SED using monochannel audio is a challenging task. These overlapping sound events can potentially be recognized better with multichannel audio.
This repository supports both single- and multichannel versions of polyphonic SED and is referred as SEDnet hereafter. You can read more about [sound event detection literature here](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-detection).

This method was first proposed in '[Sound event detection using spatial features and convolutional recurrent neural network](https://arxiv.org/abs/1706.02291 "Arxiv paper")'. It recently won the [DCASE 2017 real-life sound event detection](https://goo.gl/8eqCg3 "Challenge webpage"). We are releasing a simple vanila code without much frills here. 

If you are using anything from this repository please consider citing,

>Sharath Adavanne, Pasi Pertila and Tuomas Virtanen, "Sound event detection using spatial features and convolutional recurrent neural network" in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2017)

Similar CRNN architecture has been successfully used for different tasks and research challenges as below. You can accordingly play around with a suitable prediction layer as the task requires.

1. Sound event detection
   - Sharath Adavanne, Pasi Pertila and Tuomas Virtanen, '[Sound event detection using spatial features and convolutional recurrent neural network](https://arxiv.org/abs/1706.02291 "Arxiv paper")' at IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2017) 
   - Sharath Adavanne, Archontis Politis and Tuomas Virtanen, '[Multichannel sound event detection using 3D convolutional neural networks for learning inter-channel features](https://arxiv.org/abs/1801.09522 "Arxiv paper")' at International Joint Conference on Neural Networks (IJCNN 2018)

2. SED with weak labels
   - Sharath Adavanne and Tuomas Virtanen, '[Sound event detection using weakly labeled dataset with stacked convolutional and recurrent neural network](https://arxiv.org/abs/1710.02998 "Arxiv paper")' at Detection and Classification of Acoustic Scenes and Events (DCASE 2017)

3. Bird audio detection 
   - Sharath Adavanne, Konstantinos Drossos, Emre Cakir and Tuomas Virtanen, '[Stacked convolutional and recurrent neural networks for bird audio detection](https://arxiv.org/abs/1706.02047 "Arxiv paper")' at European Signal Processing Conference (EUSIPCO 2017)
   - Emre Cakir, Sharath Adavanne, Giambattista Parascandolo, Konstantinos Drossos and Tuomas Virtanen, '[Convolutional recurrent neural networks for bird audio detection](https://arxiv.org/abs/1703.02317 "Arxiv paper")' at European Signal Processing Conference (EUSIPCO 2017)

4. Music emotion recognition
   - Miroslav Malik, Sharath Adavanne, Konstantinos Drossos, Tuomas Virtanen, Dasa Ticha, Roman Jarina , '[Stacked convolutional and recurrent neural networks for music emotion recognition](https://arxiv.org/abs/1706.02292 "Arxiv paper")', at Sound and Music Computing Conference (SMC 2017)

## More about SEDnet
The proposed SEDnet is shown in the figure below. The input to the method is either a single or multichannel audio. The log mel-band energy feature is then extracted from each channel of the corresponding input audio. These audio features are fed to a convolutional recurrent neural network that maps them to the activities of the sound event classes in the dataset. The output of the neural network is in the continuous range of [0, 1] for each of the sound event classes and corresponds to the probability of the particular sound class being active in the frame. This continuous range output is further thresholded to obtain the final binary decision of the sound event class being active or absent in each frame. In general, the proposed method takes a sequence of frame-wise audio features as the input and predicts the activity of the target sound event classes for each of the input frames.

<p align="center">
   <img src="https://github.com/sharathadavanne/multichannel-sed-crnn/blob/master/images/CRNN_SED_DCASE2017_task3.jpg" width="400" title="SEDnet Architecture">
</p>


## Getting Started

This repository is built around the DCASE 2017 task 3 dataset, and consists of four Python scripts. 
* The feature.py script, extracts the features, labels, and normalizes the training and test split features. Make sure you update the location of the wav files, evaluation setup and folder to write features in before running it. 
* The sed.py script, loads the normalized features, and traines the SEDnet. The training stops when the error rate metric in one second segment (http://tut-arg.github.io/sed_eval/) stops improving.
* The metrics.py script, implements the core metrics from sound event detection evaluation module http://tut-arg.github.io/sed_eval/
* The utils.py script has some utility functions.

If you are only interested in the SEDnet model then just check  `get_model()` function in the sed.py script.


### Prerequisites

The requirements.txt file consists of the libraries and their versions used. The Python script is written and tested in 3.7.3 version. You can install the requirements by running the following line

```
pip install -r requirements.txt
```
## Training the SEDnet on development dataset of DCASE 2017

* Download the dataset from https://zenodo.org/record/814831#.Ws2xO3VuYUE
* Update the path of the `audio/street/` and `evaluation_setup` folders of the dataset in feature.py script. Also update the `feat_folder` variable with a local folder where the script can dump the extracted feature files. Run the script `python feature.py` this will save the features and labels of training and test splits in the provided `feat_folder`. Change the flag `is_mono` to `True` for single channel SED, and `False` for multichannel SED. Since the dataset used has only binaural audio, by setting `is_mono = False`, the SEDnet trains on binaural audio.
* In the sed.py script, update the `feat_folder` path as used in feature.py script.  Change the `is_mono` flag according to single or multichannel SED studies and run the script `python sed.py`. This should train on the default training split of the dataset, and evaluate the model on the testing split for all four folds.

The sound event detection metrics - error rate (ER) and F-score for one second segment averaged over four folds are as following. Since the dataset is small the results vary quite a bit, hence we report the mean of five separate runs. An ideal SED method has an ER of 0 and F of 1.

| SEDnet mode | ER | F|
| ----| --- | --- |
| Single channel | 0.60 | 0.57 |
| Multichannel |0.60 | 0.59|

The results vary from the original paper, as we are not using the evaluation split here

## License

This repository is licensed under the TUT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

The research leading to these results has received funding from the European Research Council under the European Unions H2020 Framework Programme through ERC Grant Agreement 637422 EVERYSOUND.
