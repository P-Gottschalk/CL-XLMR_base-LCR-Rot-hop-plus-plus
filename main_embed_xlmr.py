import argparse
import os
from typing import Optional
import xml.etree.ElementTree as ElementTree

import torch
from rdflib import Graph
from tqdm import tqdm

from model.embeddings_layer_xlmr import EmbeddingsLayerXLMR
from utils import download_from_url, EmbeddingsDataset

def get_data(year, phase, language, dirname):
    filename = f"ABSA{year % 2000}_Restaurants_{phase}_{language}.xml"

    path = f"/content/drive/MyDrive/Thesis_Data/data/{dirname}/{filename}"

    tree = ElementTree.parse(path)

    return tree

def generate_embeddings(embeddings_layer: EmbeddingsLayerXLMR, data: ElementTree, embeddings_dir: str):
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"\nGenerating embeddings into {embeddings_dir}")

    labels = {
        'negative': 0,
        'neutral': 1,
        'positive': 2,
    }

    with torch.no_grad():
        i = 0
        for node in tqdm(data.findall('.//sentence'), unit='sentence'):
            sentence = node.find('./text').text

            for opinion in node.findall('.//Opinion'):
                target_from = int(opinion.attrib['from'])
                target_to = int(opinion.attrib['to'])
                polarity = opinion.attrib['polarity']

                if polarity not in labels:
                    print(f"Unknown polarity \"{polarity}\" found at sentence \"{sentence}\"")
                    continue

                label = labels.get(polarity)

                embeddings, target_pos, hops = embeddings_layer.forward(sentence, target_from, target_to)
                data = {'label': label, 'embeddings': embeddings, 'target_pos': target_pos, 'hops': hops}
                torch.save(data, f"{embeddings_dir}/{i}.pt")
                i += 1

        print(f"Generated embeddings for {i} opinions")


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2016, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--phase", default="Train", help="The phase of the dataset (Train or Test)")
    parser.add_argument("--language", default="English", type=str, help="The language of the dataset")
    parser.add_argument("--dirname", default="processed", type=str, help= "Name of the directory of the data to be embedded.")
    parser.add_argument("--ont-hops", default=None, type=int, required=False,
                        help="The number of hops in the ontology to use")
    parser.add_argument("--vm", default=True, type=bool, action=argparse.BooleanOptionalAction,
                        help="Whether to use the visible matrix")
    parser.add_argument("--sp", default=True, type=bool, action=argparse.BooleanOptionalAction,
                        help="Whether to use soft positions")
    parser.add_argument("--all", default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help="Generate all embeddings for a given year")
    parser.add_argument("--model-spec", default = "xlm-roberta-base", type=str,help="XLM-R model that is used")
    args = parser.parse_args()

    year: int = args.year
    phase: str = args.phase
    language: str = args.language
    dirname: str = args.dirname
    ont_hops: Optional[int] = args.ont_hops
    use_vm: bool = args.vm
    use_soft_pos: bool = args.sp
    generate_all: bool = args.all
    model_spec = args.model_spec

    if ont_hops is None and (use_vm is False or use_soft_pos is False):
        raise ValueError("The visible matrix and soft positions have no effect without hops in the ontology")

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_default_device(device)

    # generate embeddings only for selected options
    if not generate_all:
        data = get_data(year, phase, language, dirname)

        embeddings_layer = EmbeddingsLayerXLMR(training = model_spec, device=device)
        embeddings_dir = EmbeddingsDataset(model = model_spec, year=year, device=device, phase=phase, language=language, ont_hops=ont_hops, empty_ok=True, use_vm=use_vm, use_soft_pos=use_soft_pos).dir
        generate_embeddings(embeddings_layer, data,embeddings_dir)
        return

    print(f"\nGenerating all embeddings for year {year}")

    for language in ['English', 'Dutch', 'French', 'Spanish']:

        for phase in ['Train', 'Test']:
            data = get_data(year, phase, language, dirname)
            embeddings_layer = EmbeddingsLayerXLMR(model_spec, device=device)
            embeddings_dir = EmbeddingsDataset(model = model_spec, year=year, device=device, phase=phase, language=language, ont_hops=None, empty_ok=True).dir
            generate_embeddings(embeddings_layer,data,embeddings_dir)


            if phase == 'Train' or 'Trial':
                continue

            print("inject knowledge for test datasets")
            for ont_hops in range(3):
                for use_vm in [True, False]:
                    for use_soft_pos in [True, False]:
                        embeddings_layer = EmbeddingsLayerXLMR(model_spec, device=device)
                        embeddings_dir = EmbeddingsDataset(model = model_spec, year=year, device=device, phase=phase, ont_hops=ont_hops,
                                                           empty_ok=True, use_vm=use_vm, use_soft_pos=use_soft_pos).dir
                        generate_embeddings(embeddings_layer,data,embeddings_dir)


if __name__ == "__main__":
    main()
