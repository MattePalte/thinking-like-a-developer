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

### More reproducibility
If you want to reproduce also the extraction of attention weights from the two neural models we share also the code we reuse from the original work plus our modifications. Follow the instruction contained in the `README_NEW_STUDY.md` file of these repositories:
1. Neural Model 1 (Convolutional-based). https://github.com/MattePalte/convolutional-attentionPrivate/blob/master/README_NEW_STUDY.md
1. Neural Model 2 (Transformer-based). https://github.com/MattePalte/NeuralCodeSumPrivate/blob/master/README_NEW_STUDY.md

For completeness we also provide you with our trained models:
1. Model 1 (Convolutional-based): https://doi.org/10.6084/m9.figshare.14414672
1. Model 2 (Transformer-based): https://doi.org/10.6084/m9.figshare.14431439

You can see our output predictions of the model as raw data here: https://doi.org/10.6084/m9.figshare.14452590.
Then to get the processed version of the model prediction in a way they are suitable for the comparative study you can use the `scripts/process_raw_human_logs.py` to preprocess raw data using respectively the functions `praparetransformer` and `prapareextsummarizer`.

## Empirical Study

To reproduce this empirical study or to deploy and run the *Human Reasoning Recorder* on your machine use.

For the empirical study, we have the following sequence of operations to apply, which are represented by jupyter notebooks, and data, which are the input and output of such steps.
1. **dataset with precomputed experiment set**: we group in a single JSON file the methods to be shown to a single participants along with all the necessary information. The dataset contains multiple of these JSON files, and it is built in such a way that the same method is proposed to 5 different users. In folder `data/datasets/methods_showed_to_original_participants` you can find the JSON files that we proposed to our participants, our survey did not deliver all those set but only the first part.
1. **Setup the Human Reasoning Recorder**: for the installation of the *Human Reasoning Recorder* you can follow the steps in the `INSTALL.md` file.
1. **processing of raw participants logs**: this downloads the data from the MongoDB database and processes them. Follow the `scripts/process_raw_human_logs.py` to download and preprocess raw data using respectively the functions `download` and `preparehuman`. For raw data of participants log of our empirical study are available here: https://doi.org/10.6084/m9.figshare.14452590.
1. **filtering**: given the number of correct answers by each participant, assign it a probability of coming from a random guesser. If this probability is high it will be rejected in the comparative study part. You can reproduce this step by following the notebook: `Filtering_Random_Guesser_Computation.ipynb`.
This is producing the `data/user_info/users_evaluation.csv` file used in the comparative study.

**OPTIONAL**: If you want to create your own dataset of methods to present to the participants you read the procedure documented in `RUN_YOUR_SURVEY.md`.





