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

- `main_clean.py`: cleans dataset by removing opinions with implicit or invalid targets which are either found in the original dataset or occur during translation and Aspect-Code-Switching.
- `main_translate.py`: contains a main method to create the datasets needed for all models which require translation. Here, we use Google API for translations. The translation process can also be carried out manually using functions.
- `main_embed_mbert.py`: generates mBERT embeddings, which are utilised when running in tuning, training and validation of our model.
- `main_embed_xlmr.py`: generates XLM-R_base and XLM-R embeddings, which are utilised when running in tuning, training and validation of our model.
- `main_hyperparam.py`: runs hyperparameter optimization. This method can be used both for models with contrastive learning and for models without contrastive learning. Note that for models with contrastive learning, we first optimise model parameters and then subsequently optimise the beta.
- `main_train.py`:  trains a specific model using the hyperparameter inputs from the tuning done in `main_hyperparam.py`.
- `main_validate.py`: validates the trained model using the test data of the requested language. Prints various model evaluation scores.
- `main_plot.py`: plots the t-SNE plots of the high-level sentiment representation vectors used in the final layer of LCR-Rot-hop++.


## Models
### XLMR_base-LCR-Rot-hop++
Clean the English training dataset using the `main_clean.py` method. Ensure that the variable "--phase" is set to "Train" and "--Language" is set to "English". Then, also use `main_clean.py` to clean the test data of the language that is used to validate the model. Here, set "--phase" to "Test". Subsequently,  create XLM-R_base embeddings, which is done with `main_embed_xlmr.py` and setting the variable "--model-spec" to "xlm-roberta-base". For XLM-R embeddings, set "--model-spec" to"xlm-roberta-large". For mBERT embeddings, use ``main\_embed\_mbert.py" instead. In all subsequent methods, ensure that the "--model-type" variable is set to the respective model type: "mBERT", "xlm-roberta-base", or "xlm-roberta-large". Hyperparameter optimisation is done through the ``main\_hyperparam.py" method. The results are saved in the "data" section, in the "checkpoints" folder. The model is trained using `main_train.py`, which requires the manual input of the hyperparameters from `main_hyperparam.py` before being run. The model can be validated using `main_validate.py`. Note "--model" needs to be set to the directory address of the model that is to be validated.

#### Adjustments for Contrastive Learning

- `main_hyperparam.py`: set the variable "--contrastive-learning" to "Sen" if to utilising the sentiment-level contrastive learning. Set the variable "--contrastive-learning" to "Rep" to utilise the contrastive learning with the concatenated representation vector from the LCR-Rot-hop++ model.
- `main_train.py`: set the variable "--contrastive-learning" as described above. Set the variable "--beta" to the beta from the checkpoints obtained by the `main_hyperparam.py` method.

### XLMR_base-LCR-Rot-hop-XX++
These models are trained using the same procedure as the XLMR_base-LCR-Rot-hop++ model, with the only change being that they are trained on embeddings from language XX (FR, ES, NL) rather than English.

### XLMR_base-MLCR-Rot-hop++
The multilingual dataset that gives  MLCR-Rot-hop++ its name is created by combining the English, French, Spanish and Dutch datasets. Either run `main_translate.py` and set the variable "--model-type" to "MLCR-Rot-hop++". Another option is to manually run the "MLCR-Rot-hop++" function in `main_translate.py`. Subsequently, follow the same steps as in the original XLMR_base-LCR-Rot-hop++, from `main_hyperparam.py` onwards.

### XLMR_base-LCR-Rot-hop-XXen++
Once again, this process can be run directly through the "main" method of `main_translate.py`. To do so, run `main_translate.py` and set the variable "--model-type" to "mLCR-Rot-hop-XXen++". The rest of the procedure (tuning, training and validating) is the same as in the other models.

To translate from English into language XX, we first need to mark the data. This can be done using the function "mark_data.py" from `main_translate.py`. Once this marking has been completed, translation can be carried out using the "translate_data" function and the symbols are removed from the dataset using the "remove_symbols" function. Lastly, the data has to be cleaned using the "clean_data" function from `main_clean.py` From this point, as above, follow the same steps as in the original XLMR_base-LCR-Rot-hop++, from `main_hyperparam.py` onwards.

### XLMR_base-Rot-hop-ACSxx
To run this model directly through `main_translate.py`, set the variable "--model-type" to "mLCR-Rot-hop-ACSxx".

To run this model manually, follow the same steps as in the XLMR_base-LCR-Rot-hop-XXen++ model to obtain the marked and the translated data, but do not use the "remove_symbols" function on the translated data. Instead, run the "aspect_code_switching" function found in "main_translate.py". This creates two new datasets which have had the aspects switched. Now, run the "remove_symbols" and "clean_data" functions on each dataset individually. The "join_datasets_ACS" function can then be used to fuse the four individual datasets. From here, continue with hyperparameter tuning as described in all other models.

### Plotting t-SNE Graphs
**Note:** This requires the models to have been trained beforehand. We also require the test data to have been embedded with the utilised embedder.

Once the respective model has been trained, the dimension-reduced (using t-SNE) high-level sentiment representation vectors can be plotted using the `main_plot.py` method. Note that we plot all languages on the same graph. To do so, the following parameters have to be set accordingly:

- "--model-type": which type of embeddings are used in the trained model ("mBERT", "xlm-roberta-base", "xlm-roberta-large")
- "--plot-name": a str with the name of the `.png` file you want the plot to have.
- "--type-plot": whether you want to distinguish plotted points via true labels (str: "label") or via predictions (str: "pred").
- "--model-base": directory address of the base model (for XLMR_base-LCR-Rot-hop++ and XLMR_base-MLCR-Rot-hop++) or the English model (XLMR_base-LCR-Rot-hop-XX++). Leave empty for XLMR_base-LCR-Rot-hop-XXen++ and XLMR_base-Rot-hop-ACSxx.
- "--model-dutch": directory address for Dutch model (XLMR_base-LCR-Rot-hop-XX++, XLMR_base-LCR-Rot-hop-XXen++, XLMR_base-Rot-hop-ACSxx). Leave empty for XLMR_base-LCR-Rot-hop++ and XLMR_base-MLCR-Rot-hop++. Note that this will only be plotted if directory addresses are also given for the French and Spanish models.
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
2. `Shell_Multilingual.ipynb`: run the "Hyperparameter Optimisation" and "Model Training" subsections of "XLMR-LCR-Rot-hop++". Note that the optimal hyperparameters have to be manually inputted for "Model Training". This returns the mLCR-Ror-hop++, XLMR_base-LCR-Rot-hop++ and XLMR-LCR-Rot-hop++ model. To validate the results for the English language, also run the "Model Validation" subsection. Here, the directory address of the model needs to be manually inputted.
3. `Shell_ _Language_ _Tuning_Training_Validation.ipynb`: to validate the results on _Language_, run the "Model Validation" subsection of the "XLMR-LCR-Rot-hop++" section.

### XLMR_base-LCR-Rot-hop-XX++

1. `Shell_Clean_Embedding.ipynb`: run the "Cleaning" section of the .ipynb file. Then, run the "_Language_ Embeddings" subsection in "Embeddings", corresponding to language XX, excluding the three statements related to the ACS_XX methodology.
2. `Shell_ _Language_ _Tuning_Training_Validation.ipynb`: run the "Hyperparameter Optimisation" and "Model Training" subsections of "XLMR-LCR-Rot-hop++". Note that the optimal hyperparameters have to be manually inputted for "Model Training". To validate the results on _Language_, run the "Model Validation" subsection of the "XLMR-LCR-Rot-hop++" section. As above, the directory address of the model requires manual input.

### XLMR_base-MLCR-Rot-hop++

1. `Shell_Clean_Embedding.ipynb`: run the "Cleaning" section of the .ipynb file.
2. `Shell_Translate.ipynb`: run the "XLMR_base-MLCR-Rot-hop++" section of the file.
3. `Shell_Clean_Embedding.ipynb`:  run the "Multilingual Embeddings" subsection in "Embeddings". Also run the embeddings subsection of any language used for model testing.
4. `Shell_Multilingual.ipynb`: run the "Hyperparameter Optimisation" and "Model Training" subsections of "XLMR-MLCR-Rot-hop++". Note that the optimal hyperparameters have to be manually inputted for "Model Training". This returns the MLCR-Rot-hop++, XLMR_base-MLCR-Rot-hop++ and XLMR-MLCR-Rot-hop++ model. To validate the results for the English language, also run the "Model Validation" subsection. Here, the directory address of the model needs to be manually inputted.
5. `Shell_ _Language_ _Tuning_Training_Validation.ipynb`: to validate the results on _Language_, run the "Model Validation" subsection of the "XLMR-MLCR-Rot-hop++" section.

### XLMR_base-Rot-hop-ACSxx

1. `Shell_Translate.ipynb`: run the statement corresponding to language XX in the XLMR_base-LCR-Rot-hop-ACSxx section of the file.
2. `Shell_Clean_Embedding.ipynb`:  for language XX, run the three statements related to the ACS_XX methodology in the "_Language_ Embeddings" subsection, found in the larger "Embeddings" section.
3. `Shell_ _Language_ _Tuning_Training_Validation.ipynb`: run the "Hyperparameter Optimisation" and "Model Training" subsections of "XLMR-LCR-Rot-hop-ACS_XX++". Note that the optimal hyperparameters have to be manually inputted for "Model Training". To validate the results on _Language_, run the "Model Validation" subsection of the "XLMR-LCR-Rot-hop-ACS_XX++" section. The directory address of the model requires manual input.

### Contrastive Learning: All Models

All contrastive learning models (CLS- and CLR-) can be run in a similar manner as the models described above. Until the optimisation of the hyperparameters, the steps that need to be taken are identical. From here, run the "Hyperparameter Optimisation", "Model Training" and "Model Validation" in the respective CLS- and CLR- sections of the respective documents.

### Plotting: All Models

1. `Shell_Plot.ipynb`: Once the respective models have been trained, and the test data of all languages has been embedded, simply run the required subsection ("Base Models", "CLS Models", "CLR Models") in the respective model section of the notebook.

## Acknowledgements
This repository is based on the source code of https://github.com/Anonymous71717/mLCR-Rot-hop-plus-plus.git.

The `model.bert_encoder` module uses code from:

- Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., Wang, P.: K-BERT: Enabling language representation with
  knowledge graph. In: 34th AAAI Conference on Artificial Intelligence. vol. 34, pp. 2901–2908. AAAI Press (2020)
- https://github.com/Felix0161/KnowledgeEnhancedABSA

The `utils.sent_con_loss module` is based on code from:

- Lin, N., Fu, Y., Lin, X., Zhou, D., Yang, A. & Jiang, S. (2023). CL-XABSA: contrastive learning for cross-lingual aspect-based sentiment analysis. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31 , 2935-2946.


