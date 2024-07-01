# Partially inspired by https://github.com/GKLMIP/CL-XABSA.git

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne(data, labels, dir_name: str, plot_name: str):
    tsne = TSNE(n_components=2, random_state=42)

    print("Running t-SNE...")
    language_tsne = tsne.fit_transform(data)
    print("t-SNE is finished. Plotting results...")

    pos_reduced_x, pos_reduced_y = [],[]
    neu_reduced_x, neu_reduced_y = [],[]
    neg_reduced_x, neg_reduced_y = [],[]

    for i in range(len(labels)):
        if labels[i] == 2:
            pos_reduced_x.append(language_tsne[i][0])
            pos_reduced_y.append(language_tsne[i][1])
        elif labels[i] == 1:
            neu_reduced_x.append(language_tsne[i][0])
            neu_reduced_y.append(language_tsne[i][1])
        elif labels[i] == 0:
            neg_reduced_x.append(language_tsne[i][0])
            neg_reduced_y.append(language_tsne[i][1])

    plt.figure(figsize=(8,8))

    plt.scatter(pos_reduced_x, pos_reduced_y, c='green', marker='.')
    plt.scatter(neu_reduced_x, neu_reduced_y, c='grey', marker='.')
    plt.scatter(neg_reduced_x, neg_reduced_y, c='red', marker='.')

    plt.legend(labels=["Positive", "Neutral", "Negative"],loc="upper right",fontsize=10).get_texts()


    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')

    plt.savefig(f"{dir_name}/{plot_name}",format = "png")
    print(f"\nSaved plot to {dir_name}/{plot_name}.png")

def get_results(model:LCRRotHopPlusPlus, language:str, dataset: EmbeddingsDataset, type_plot: str):
    test_loader = DataLoader(dataset, collate_fn=lambda batch: batch)

    print(f"Add {language} embeddings to t-SNE plot.")

    collection = []
    sentiments = []

    for i, data in enumerate(tqdm(test_loader, unit='obs')):
        torch.set_default_device(dataset.device)

        with torch.no_grad():
            (left, target, right), label, hops = data[0]

            if target is not None and len(target) != 0:
                sent_output,rep_output = model(left, target, right, hops)
                sent_label = sent_output.argmax(0)

                collection.append(rep_output)
                
                if type_plot == "label":
                    sentiments.append(label)
                elif type_plot == "pred":
                    sentiments.append(sent_label)

            else:
                print(f"\n[Invalid instance in batch {i}. Instance skipped]")

        torch.set_default_device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    torch.cuda.empty_cache()

    return collection,sentiments

def main():

    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-type", default = "mBERT", type = str, help = "Model that is trained")
    parser.add_argument("--plot-name", default="my_plot", type=str, help="Name of the plot to save")
    parser.add_argument("--year", default=2016, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--phase", default="Test", help="The phase of the dataset (Train, Test or Trial)")
    parser.add_argument("--ont-hops", default=None, type=int, required=False, help="The number of hops in the ontology")
    parser.add_argument("--hops", default=3, type=int,
                        help="The number of hops to use in the rotatory attention mechanism")
    parser.add_argument("--gamma", default=None, type=int, required=False,
                        help="The value of gamma for the LCRRotHopPlusPlus model")
    parser.add_argument("--vm", default=True, type=bool, action=argparse.BooleanOptionalAction,
                        help="Whether to use the visible matrix")
    parser.add_argument("--sp", default=True, type=bool, action=argparse.BooleanOptionalAction,
                        help="Whether to use soft positions")
    parser.add_argument("--type-plot", default="label", type = str, 
                        help = "Choose which to plot: label or pred")
    
    parser.add_argument("--model-base",  type=str, required=False, help="Path to a state_dict of the LCRRotHopPlusPlus model")
    parser.add_argument("--model-dutch", type=str, required=False, help="Path to a state_dict of a Dutch LCRRotHopPlusPlus model")
    parser.add_argument("--model-french", type=str, required=False, help="Path to a state_dict of a French LCRRotHopPlusPlus model")
    parser.add_argument("--model-spanish", type=str, required=False, help="Path to a state_dict of a Spanish LCRRotHopPlusPlus model")

    args = parser.parse_args()

    model_type: str = args.model_type
    year: int = args.year
    phase: int = args.phase
    ont_hops: Optional[int] = args.ont_hops
    hops: int = args.hops
    gamma: Optional[int] = args.gamma
    use_vm: bool = args.vm
    use_soft_pos: bool = args.sp
    plot_name = args.plot_name
    type_plot = args.type_plot
    
    model_base: Optional[int] = args.model_base
    model_dutch: Optional[int] = args.model_dutch
    model_french: Optional[int] = args.model_french
    model_spanish: Optional[int] = args.model_spanish

    if type_plot == "label":
        dir_path = "/content/drive/MyDrive/Thesis_Data/data/plots_label"
    elif type_plot == "pred":
        dir_path = "/content/drive/MyDrive/Thesis_Data/data/plots_pred"
    else:
        raise ValueError("Invalid type of plotting.")

    os.makedirs(dir_path, exist_ok=True)

    if model_base is None:
        print("English language data not included in this model.")
    else:
        print("English language data is included in this model.")

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    
    english_data = EmbeddingsDataset(model=model_type, year=year, device=device, phase=phase, language="English", ont_hops=ont_hops, use_vm=use_vm,
                                    use_soft_pos=use_soft_pos)
    dutch_data = EmbeddingsDataset(model=model_type, year=year, device=device, phase=phase, language="Dutch", ont_hops=ont_hops, use_vm=use_vm,
                                    use_soft_pos=use_soft_pos)
    french_data = EmbeddingsDataset(model=model_type, year=year, device=device, phase=phase, language="Spanish", ont_hops=ont_hops, use_vm=use_vm,
                                    use_soft_pos=use_soft_pos)
    spanish_data = EmbeddingsDataset(model=model_type, year=year, device=device, phase=phase, language="French", ont_hops=ont_hops, use_vm=use_vm,
                                    use_soft_pos=use_soft_pos)
    
    input_size = 1024 if model_type == "xlm-roberta-large" else  768
    
    # distinguish between language-agnostic and specific language models
    if model_dutch is None or model_french is None or model_spanish is None:
        model = LCRRotHopPlusPlus(input_size=input_size, gamma=gamma, hops=hops).to(device)
        state_dict = torch.load(model_base, map_location=device)
        model.load_state_dict(state_dict)

        model.eval()

        english_results, english_labels = get_results(model=model, language="English",dataset=english_data, type_plot=type_plot)
        dutch_results, dutch_labels= get_results(model=model, language="Dutch",dataset=dutch_data, type_plot=type_plot)
        french_results, french_labels= get_results(model=model, language="French",dataset=french_data, type_plot=type_plot)
        spanish_results, spanish_labels = get_results(model=model, language="Spanish",dataset=spanish_data, type_plot=type_plot)
    else:
        model_nl = LCRRotHopPlusPlus(input_size=input_size, gamma=gamma, hops=hops).to(device)
        model_fr = LCRRotHopPlusPlus(input_size=input_size, gamma=gamma, hops=hops).to(device)
        model_es = LCRRotHopPlusPlus(input_size=input_size, gamma=gamma, hops=hops).to(device)

        state_dict_nl = torch.load(model_dutch, map_location=device)
        state_dict_fr = torch.load(model_french, map_location=device)
        state_dict_es = torch.load(model_spanish, map_location=device)

        model_nl.load_state_dict(state_dict_nl)
        model_fr.load_state_dict(state_dict_fr)
        model_es.load_state_dict(state_dict_es)

        model_nl.eval()
        model_fr.eval()
        model_es.eval()

        dutch_results, dutch_labels = get_results(model=model_nl, language="Dutch",dataset=dutch_data, type_plot=type_plot)
        french_results, french_labels = get_results(model=model_fr, language="French",dataset=french_data, type_plot=type_plot)
        spanish_results, spanish_labels = get_results(model=model_es, language="Spanish",dataset=spanish_data, type_plot=type_plot)

        if model_base is not None:
            model = LCRRotHopPlusPlus(input_size=input_size, gamma=gamma, hops=hops).to(device)
            state_dict = torch.load(model_base, map_location=device)
            model.load_state_dict(state_dict)

            model.eval()

            english_results, english_labels = get_results(model=model, language="English",dataset=english_data, type_plot=type_plot)
        else:
            english_results, english_labels = [],[]


    results = english_results + dutch_results + french_results + spanish_results
    results = torch.stack(results, dim = 0)

    labels = english_labels + dutch_labels + french_labels + spanish_labels

    results_np = results.cpu().numpy()

    tsne(data=results_np,labels = labels,dir_name=dir_path,plot_name=plot_name)


if __name__ == "__main__":
    main()