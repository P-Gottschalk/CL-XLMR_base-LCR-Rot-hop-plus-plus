# CLS-XMLR_base-LCR-Rot-hop++

Source code for a neural approach with hierarchical rotary attention for Multilingual- and Cross-lingual Aspect-Based Sentiment
Classification using Contrastive Learning.

## Data

First, create a `data/raw` directory and download
the [SemEval 2016](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)
dataset for the language to test. Note here that we use the Subtask 1 data for respective languages. Then rename the SemEval datasets to end up with the following files:

- `data/raw`
    - `ABSA16_Restaurants_Test_Language.xml`
    - `ABSA16_Restaurants_Train_Language.xml`

Note that this directory addresses may have to be adapted in the code itself, as the directories are currently set for a combined usage of Google Colab and Google Drive. The current form in the Colab optimised code is as follows: `/content/drive/MyDrive/Thesis_Data/data/raw`

## Usage

To view the available cli args for a program, run `python [FILE] --help`. These CLI args can be used to modify varioys aspects of the code which will be run. 

Note that the some files include an ontology reasoner, and an ontology injection method, which can be used to enhance the final performance of the models. This research did not utilize an ontology reasoner nor the injection method. For descriptions to use this method, see https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus.git.

- `main_clean.py`: remove opinions that contain implicit targets and invalid targets due to translation or Aspect-Code-Switching
- `main_translate.py`: contains a main method to create the datasets needed for all models which require translation. Also contains all functions needed to create Multilingual datasets manually, a description of how to run each model is given below. Our version uses Google API for translation. 
- `main_embed_mbert.py`: generate mBERT embeddings, these embeddings are used by the other programs.
- `main_embed_xlmr.py`: generate XLM-R_base and XLM-R embeddings, these embeddings are used by the other programs.
- `main_hyperparam.py`: run hyperparameter optimization. This method can be used both for models with contrastive learning and for models without contrastive learning. Note that for models with contrastive learning, we first optimise model parameters and then subsequently optimise the beta.
- `main_train.py`: train the model for a given set of hyperparameters.
- `main_validate.py`: validate a trained model.

## Models
### XLMR_base-LCR-Rot-hop++
Run `main_clean.py` on the English train and test datasets, which is selected by passing either "Train" or "Test" as parameter inputs for the variable "phase". Note that the test dataset for the language that the model is evaluated on also needs to be cleaned. Second, create XLM-R_base embeddings, which is done with `main_embed_xlmr.py` and setting the variable "--model-spec" to "xlm-roberta-base". For XLM-R embeddings, set "--model-spec" to "xlm-roberta-large". For mBERT embeddings, use `main_embed_mbert.py` instead. Then, run `main_hyperparam.py`, which provides a checkpoint file in the "data" folder, which contains the hyperparameter values needed to be changed in `main_train.py`. After updating the hyperparameters manually, run `main_train.py`, which gives a model as output, ready to be tested. Last, validate the model with `main_validate.py`, which outputs the performance measures.

#### Adjustments for Contrastive Learning

- `main_hyperparam.py`: set the variable "--contrastive-learning" to "Sen" if to utilising the sentiment-level contrastive learning. Set the variable "--contrastive-learning" to "Rep" to utilise the contrastive learning with the concatenated representation vector from the LCR-Rot-hop++ model.
- `main_train.py`: set the variable "--contrastive-learning" as described above. Set the variable "--beta" to the beta from the checkpoints obtained the `main_hyperparam.py` method.

### XLMR_base-LCR-Rot-hop-XX++
For these models, the same procedure as in XLMR_base-LCR-Rot-hop++ can be followed. However, the data corresponding to the language XX should be used for training and testing.

### XLMR_base-MLCR-Rot-hop++
To run MLCR-Rot-hop++, the English, Dutch, French, Spanish cleaned training data files have to be combined. Either run `main_translate.py` and set the variable "--model-type" to "MLCR-Rot-hop++". Another option is to manually run the "MLCR-Rot-hop++" function in `main_translate.py`. After combining all into the multilingual dataset, the rest of the procedure is the same as for mLCR-Rot-hop++.

### XLMR_base-LCR-Rot-hop-XXen++
Once again, this process can be run directly through the "main" method of `main_translate.py`. To do so, run `main_translate.py` and set the variable "--model-type" to "mLCR-Rot-hop-XXen++". The rest of the procedure (tuning, training and validating) is the same as in the other models.

To run this model manually, note that translation is necessary. Hence, the English training data first needs to be marked with the `mark_data.py` function in `main_translate.py`. Then the translation is done by running the `translate_data.py` function and specifying the target language as an input parameter. After running the translations, the markings need to be removed. That is done by the function "remove_symbols". After that, the file has to be cleaned again by `main_clean.py` to remove opinions for which the translation failed. Then it can be run the same as the other models above.

### XLMR_base-Rot-hop-ACSxx
To run this model directly through `main_translate.py`, set the variable "--model-type" to "mLCR-Rot-hop-ACSxx".

To run the model manually, which implements Aspect-Code-Switching, the same marked data created in the process for mLCR-Rot-hop-XXen and translated data, with markings, are used. The function Aspect-Code-Switching needs to be called and creates two more datasets with markings. Then all markings need to be removed per dataset with the function "remove_symbols". Afterwards, all files need to be cleaned again to remove any failed switches. Then the function "join_datasets_ACS" needs to be called to combine the four datasets for one ACS model with xx as target language.

### Plotting t-SNE Graphs
**Note:** This requires the models to have been trained beforehand. We also require the test data to have been embedded with the utilised embedder.

Once the respective model has been trained, the dimension-reduced (using t-SNE) high-level sentiment representation vectors can be plotted using the `main_plot.py` method. Note that we plot all languages on the same graph. To do so, the following parameters have to be set accordingly:

- "--model-type": which type of embeddings are used in the trained model ("mBERT", "xlm-roberta-base", "xlm-roberta-large")
- "--plot-name": a str with the name of the `.png` file you want the plot to have.
- "--type-plot": whether you want to distinguish plotted points via true labels (str: "label") or via predictions (str: "pred").
- "--model-base": directory address of the base model (for XLMR_base-LCR-Rot-hop++ and XLMR_base-MLCR-Rot-hop++) or the english model (XLMR_base-LCR-Rot-hop-XX++). Leave empty for XLMR_base-LCR-Rot-hop-XXen++ and XLMR_base-Rot-hop-ACSxx.
- "--model-dutch": directory address for Dutch model (XLMR_base-LCR-Rot-hop-XX++, XLMR_base-LCR-Rot-hop-XXen++, XLMR_base-Rot-hop-ACSxx). Leave empty for XLMR_base-LCR-Rot-hop++ and XLMR_base-MLCR-Rot-hop++. Note that this will only be plotted if directory addresses are also given for the French and Spanish model.
- "--model-french": as above.
- "--model-spanish": as above. 

## Run with Google Colab
Note: this entire section is optional to use. If you prefer not to run in Google Colab, you can ignore this section.

To ease the running of the code found within this repository, shell notebooks in the form of .ipynb files have been created to deal with respective tasks. Since our paper utilises English, French, Spanish and Dutch data, we have shells for the usage of these languages. Whilst it is possible to run an XLMR_base-LCR-Rot-hop-XXen++ model using this repository, as this was not treated in our paper, no Shell has been implemented. The following notebooks are available for usage, and can be found in the `Shell` folder of this GitHub.

- `Shell_Clean_Embedding.ipynb`: used for cleaning and embedding of the data in various languages.
- `Shell_Translation.ipynb`: used for the creation of multilingual and translated datasets. Needed for XLMR_base-MLCR-Rot-hop++ and XLMR_base-Rot-hop-ACSxx models.
- `Shell_Multilingual.ipynb`: used for the tuning/training of multilingual models. Also used for the tuning and training of the XLMR_base-LCR-Rot-hop++ model, which is trained on the English dataset. Validation of the English dataset is also carried out in this Shell.
- `Shell_French_Tuning_Training_Validation.ipynb`: used for the tuning, training, and validation of the models which are tested using French language data.
- `Shell_Spanish_Tuning_Training_Validation.ipynb`: used for the tuning, training, and validation of the models which are tested using Spanish language data.
- `Shell_Dutch_Tuning_Training_Validation.ipynb`: used for the tuning, training, and validation of the models which are tested using Dutch language data.
- `Shell_Plot.ipynb`: used for the plotting of t-SNE graphs once the models have been trained.

To store data, create a folder within Google Drive. Ensure you change the directory addresses within the code of the GitHub to match your created folders.

### XLMR_base-LCR-Rot-hop++

1. `Shell_Clean_Embedding.ipynb`: run the "Cleaning" section of the .ipynb file. Then, run the "English Embeddings" subsection in "Embeddings". Also run the embeddings subsection of any language used for model testing.
2. `Shell Multilingual.ipynb`: run the "Hyperparameter Optimisation" and "Model Training" subsections of "XLMR-LCR-Rot-hop++". Note that the optimal hyperparameters have to be manually inputted for "Model Training". This returns the mLCR-Ror-hop++, XLMR_base-LCR-Rot-hop++ and XLMR-LCR-Rot-hop++ model. To validate the results for the English language, also run the "Model Validation" subsection. Here, the directory address of the model needs to be manually inputted.
3. `Shell_ _Language_ _Tuning_Training_Validation.ipynb`: to validate the results on _Language_, run the "Model Validation" subsection of the "XLMR-LCR-Rot-hop++" section.

### XLMR_base-LCR-Rot-hop-XX++

1. `Shell_Clean_Embedding.ipynb`: run the "Cleaning" section of the .ipynb file. Then, run the "_Language_ Embeddings" subsection in "Embeddings", corresponding to language XX, excluding the three statements related to the ACS_XX methodology.
2. `Shell_ _Language_ _Tuning_Training_Validation.ipynb`: run the "Hyperparameter Optimisation" and "Model Training" subsections of "XLMR-LCR-Rot-hop++". Note that the optimal hyperparameters have to be manually inputted for "Model Training". To validate the results on _Language_, run the "Model Validation" subsection of the "XLMR-LCR-Rot-hop++" section. As above, the directory address of the model requires manual input.

### XLMR_base-MLCR-Rot-hop++

1. `Shell_Clean_Embedding.ipynb`: run the "Cleaning" section of the .ipynb file.
2. `Shell_Translate.ipynb`: run the XLMR_base-MLCR-Rot-hop++ section of the file.
3. `Shell_Clean_Embedding.ipynb`:  run the "Multilingual Embeddings" subsection in "Embeddings". Also run the embeddings subsection of any language used for model testing.
4. `Shell Multilingual.ipynb`: run the "Hyperparameter Optimisation" and "Model Training" subsections of "XLMR-MLCR-Rot-hop++". Note that the optimal hyperparameters have to be manually inputted for "Model Training". This returns the MLCR-Ror-hop++, XLMR_base-MLCR-Rot-hop++ and XLMR-MLCR-Rot-hop++ model. To validate the results for the English language, also run the "Model Validation" subsection. Here, the directory address of the model needs to be manually inputted.
5. `Shell_ _Language_ _Tuning_Training_Validation.ipynb`: to validate the results on _Language_, run the "Model Validation" subsection of the "XLMR-MLCR-Rot-hop++" section.

### XLMR_base-Rot-hop-ACSxx

1. `Shell_Translate.ipynb`: run the statement corresponding to language XX in the XLMR_base-LCR-Rot-hop-ACSxx section of the file.
2. `Shell_Clean_Embedding.ipynb`:  for language XX, run the the three statements related to the ACS_XX methodology in the "_Language_ Embeddings" subsection, found in the larger "Embeddings" section.
3. `Shell_ _Language_ _Tuning_Training_Validation.ipynb`: run the "Hyperparameter Optimisation" and "Model Training" subsections of "XLMR-LCR-Rot-hop-ACS_XX++". Note that the optimal hyperparameters have to be manually inputted for "Model Training". To validate the results on _Language_, run the "Model Validation" subsection of the "XLMR-LCR-Rot-hop-ACS_XX++" section. The directory address of the model requires manual input.

### Contrastive Learning: All models

All contrastive learning models (CLS- and CLR-) can be run in a similar manner as the models described above. Until the the optimisation of the hyperparameters, the steps that need to be taken are identical. From here, run the "Hyperparameter Optimisation", "Model Training" and "Model Validation" in the respective CLS- and CLR- sections of the respective documents.

### Plotting: All Models

1. `Shell_Plot.ipynb`: Once the respective models have been trained, and the test data of all languages has been embedded, simply run the required subsection ("Base Models", "CLS Models", "CLR MOdels") in the respective model section of the notebook.

## Acknowledgements
This repository is based on the source code of https://github.com/Anonymous71717/mLCR-Rot-hop-plus-plus.git.

The `model.bert_encoder` module uses code from:

- Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., Wang, P.: K-BERT: Enabling language representation with
  knowledge graph. In: 34th AAAI Conference on Artificial Intelligence. vol. 34, pp. 2901–2908. AAAI Press (2020)
- https://github.com/Felix0161/KnowledgeEnhancedABSA

The `utils.sent_con_loss module` is based on code from:

- Lin, N., Fu, Y., Lin, X., Zhou, D., Yang, A. & Jiang, S. (2023). CL-XABSA: contrastive learning for cross-lingual aspect-based sentiment analysis. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31 , 2935-2946.


