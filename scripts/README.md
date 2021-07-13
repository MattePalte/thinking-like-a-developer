# How to use the `process_raw_human_logs.py` script

You can download the raw data directly via:
https://doi.org/10.6084/m9.figshare.14452590.

Run the following commands:

## Convolutional-based
1. Models trained (one for each of the ten projects)
1. Models predictions and weights extraction (one for each of the ten projects, follow the indication in our Ext. Summarizer repository version). You can download them directly via the link:
https://doi.org/10.6084/m9.figshare.14414672
Note that there is a set of attention weights for each predicted tokens.
1. Condense (using the average) the attention weights to have one set of attention weights for each single method. This step will also keep only methods relevant for which we have human annotations:
    - Move the content of **extracted_attention_weights** folder (just downloaded) in **data/raw_data/extreme_summarizer/** folder.
    - To convert the raw data into machine attention weights run this from the
    **scripts** folder:
    ```console
    python3 process_raw_human_logs.py prapareextsummarizer
    ```
    *WARNING: this is a lengthy process since it will condense all the methods
    in the dataset. It takes around 9 hours on a laptop with i7 processor. So we suggest downloading the preprocessed data directly.*

## Transformer-based
1. Models trained (one for each of the ten projects)
1. Models predictions and weights extraction (one for each of the ten projects, follow the indication in our Ext. Summarizer repository version). You can download them directly via the link:
https://doi.org/10.6084/m9.figshare.14431439
Note that here we already have a single set of attention weights for each method. And we already work only on the methods for which we have human annotations. The only step to perform is to condense the eight attention heads into one (using the averaging).
1. Create a single file containing all the machine attention data:
    - Move the content of **tmp_single_project** folder (just downloaded) in **data/raw_data/transformer/** folder. Note that the big files with *mdl* and *mdl.checkpoint* extensions are not needed.
    - To convert the raw data into machine attention weights run this from the
    **scripts** folder:
    ```console
    python3 process_raw_human_logs.py praparetransformer
    ```

# Repository Content when cloned
When you download the repository you find the following content in the data folder:
data
```
└─── README.md
│
└─── raw_data
|    └─── empty folder
└─── datasets
|    └─── methods_showed_to_original_participants
|           └─── ... .json|
└─── precomputed_functions_info
     └─── functions_sampled_for_the_experiment.json
```

# Preprocessed files

You can download the preprocessed files ready for analysis directly from:
https://doi.org/10.6084/m9.figshare.14462052

They contain:
```
data
└─── README.md
│
└─── processed_human_attention
|    └─── attentions_processed_also_rejected_users.json
│    └─── users_processed_also_rejected_users.json
|
└─── user_info
|    └─── users_evaluation.csv
│    └─── users_provenance.csv
│
└─── precomputed_model_prediction
     └─── extreme_summarizer
     |    └─── machine_attention.json
     └─── transformer
          └─── machine_attention.json
```

# Steps to Prepare the Data

The raw data collected during the experiment (in the same format in which they were stored in MongoDB) can be directly download via the following link:


To convert the raw data into human attention weights:

```console
python3 process_raw_human_logs.py preparehuman --include_rejected_users True
```

## Running your own experiment
In case you are using the setup for a new experiment you have to configure the **config/setting.yaml** file to point to the right mongodb database and then run the script to download the data from withing the **script** folder:
```console
python3 process_raw_human_logs.py download
```
The output will be stored in the **data/raw_data** folder.