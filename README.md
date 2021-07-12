# Thinking Like a Developer

Related research publication:
"Thinking Like a Developer? Comparing the Attention of Humans with Neural Models of Code", *conditionally accepted at ASE '21*

## Reproducibility Levels and Target Audience
This publication involves two level of reproducibility:
1. **Comparative study neural models versus human attention**: we compare the attention weights of two neural models with the collected human data. At this level, you can use the preprocessed human attention data from participants and the precomputed models' attention weights and re-create the figures of the paper.

    **Target audience:** <u>Researchers/Practitioners developing a new Software Analytics approach</u>:
    You can benchmark your approach form an explainability perspective by analyzing whether the explainability metric of your model (e.g. attention weights or feature importance) is correlated to our human attention dataset. This could help you highlight strength and weaknesses of your neural model.

1. **Empirical study**: we collect data on how humans inspect source code while performing the method naming task via the *Human Reasoning Recorder* (HRR) interface. At this deeper level, we explain you how to set up the *Human Reasoning Recorder* interface, prepare the methods to show and how to process the human logs into an attention score.

    **Target audience:** <u>Software Engineering Researchers performing a user study with developers</u>:

    You can use our *Human Reasoning Recorder* web interface to submit a series of task to a crowd of participants (remote participation allowed, such as Amazon Mechanical Turk) and collect the results on how developers perform the task. The web environment let the participants inspect only code under the mouse pointer such that the mouse trace can approximate the attention of the human, in a similar way to use of fixation time in expensive eye-tracking experiment.


## Comparative Study: Neural models versus human attention

To reproduce this study, follow these steps:
1. clone this repo;
1. make sure python 3 is installed and you have the dependencies listed in `requirements.txt` (we suggest to use a virtual environment of your choice);
1. download the data and place the unzipped content in the `data` folder of this repo: https://doi.org/10.6084/m9.figshare.14462052
1. open the `notebooks` folder and run the notebook `Comparative_Study.ipynb` to re-run the comparative study and recreate also the figures.

More reproducibility: if you want to reproduce also the extraction of attention weights from the two neural models we share also the code we reuse from the original work plus our modifications. Follow the instruction contained in the `README_NEW_STUDY.md` file of these repositories:
1. Neural Model 1: Convolutional-based. https://github.com/MattePalte/convolutional-attentionPrivate/blob/master/README_NEW_STUDY.md
1. Neural Model 2: Transformer-based. https://github.com/MattePalte/NeuralCodeSumPrivate/blob/master/README_NEW_STUDY.md

For completeness we also provide you with our trained models.


## Empirical Study

To reproduce this empirical study or to deploy and run the Human Reasoning Recorder on your machine use.



