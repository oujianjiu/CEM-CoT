# Interpreting Language Models via Concept Bottleneck Embedding in NLP tasks

## Task 1: Concept Embedding Large Language Models for Text Classification

### Setup
Recommend using cuda12.1, python3.10, pytorch2.2.
Install the packages:
```
pip install -r requirements.txt
```
### Training
#### Concept Alignment Score (CAS)
```
python label_concepts.py
```

#### Train CBL
To train the CBL, run
```
python train_CELLM.py
```

#### Train the final predictor
To train the final predictor, run
```
python train_FinalLayer.py --cbl_path */CBL.pt
```
model saved as CBL.pt in the slected path.

#### Train the baseline black-box model
```
python backbox_model.py
```

#### Test CE-LLM-TC
To test the accuracy of the CE-LLM-TC, run
```
python test_CELLM.py --cbl_path */CBL.pt
```
Please change the argument `--cbl_path` accordingly if using other settings.

#### Generate explanations from CB-LLM
Visualize the neurons in CE-LLM_TC, run
```
python CAS_neu_activ.py --cbl_path */cbl.pt
```
It generates 5 most related samples for each neuron explanation.

#### Test
To test the accuracy of the baseline standard black-box model, run
```
python test_pretrainLLM.py --model_path */backbone_finetuned_ag_news.pt
```

## Part II: CB-LLM (generation)
### Setup
Recommend using cuda12.1, python3.10, pytorch2.2.
Install the packages:
```
pip install -r requirements.txt
```
### Training
#### Train CB-LLM (generation)
To train the CB-LLM for text generation, run
```
python train_model.py
```
This will train the CE-LLM (Lora finetune Llama3 with CBL) on SST2 dataset with the class labels as concepts (negative or positive), and store the model under `from_pretained_llama3_lora_cbm/SetFit_sst2/`.
Set the argument `--dataset yelp_polarity`, `--dataset ag_news`, or `--dataset dbpedia_14` to switch the dataset.
### Testing
#### Test the concept detection of CB-LLM (generation)
To test the concept detection (concept accuracy) of the CB-LLM, run
```
python concepts_ev.py
```

#### Test the steerability of CB-LLM-CoT
```
python train_labelClasses.py
```
Set the argument `--dataset yelp_polarity`, `--dataset ag_news`, or `--dataset dbpedia_14` to switch the dataset.
After getting the classifier corresponding to the dataset, evaluate the steerability by running
```
python eva_steerability.py
```
Please rename the desired checkpoint of the peft model and CBL as `llama3` and `cbl.pt`, as the script recognizes these file names.

#### using the concept neurons and guide a sentence generation using CB-LLM-CoT
Run
```
python test_generation.py
```

