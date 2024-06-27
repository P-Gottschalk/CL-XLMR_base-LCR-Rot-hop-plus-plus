import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, CSVWriter


def validate_model(model: LCRRotHopPlusPlus, dataset: EmbeddingsDataset, name='LCR-Rot-hop++'):
    test_loader = DataLoader(dataset, collate_fn=lambda batch: batch)

    print(f"Validating model using embeddings from {dataset}")

    # run validation
    n_classes = 3
    n_correct = [0 for _ in range(n_classes)]
    n_label = [0 for _ in range(n_classes)]
    n_predicted = [0 for _ in range(n_classes)]
    brier_score = 0

    for i, data in enumerate(tqdm(test_loader, unit='obs')):
        torch.set_default_device(dataset.device)

        with torch.no_grad():
            (left, target, right), label, hops = data[0]

            if target is not None and len(target) != 0:
                output,_ = model(left, target, right, hops)
                pred = output.argmax(0)
                is_correct: bool = (pred == label).item()

                n_label[label.item()] += 1
                n_predicted[pred.item()] += 1

                if is_correct:
                    n_correct[label.item()] += 1

                for j in range(n_classes):
                    if (j == label).item():
                        brier_check = 1
                    else:
                        brier_check = 0

                    p: float = output[j].item()
                    brier_score += (p - brier_check) ** 2
            else:
                print(f"\n[Invalid instance in test batch {i}. Instance skipped]")

        torch.set_default_device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    precision = 0
    recall = 0
    for i in range(n_classes):
        if not n_predicted[i] == 0:
            precision += (n_correct[i] / n_predicted[i]) / n_classes
        recall += (n_correct[i] / n_label[i]) / n_classes

    perf_measures = {
        'Model': name,
        'Correct': sum(n_correct),
        'Accuracy': f'{(sum(n_correct) / sum(n_label)) * 100:.2f}%',
        'Precision': f'{precision * 100:.2f}%',
        'Recall': f'{recall * 100:.2f}%',
        'F1-score': f'{(2 * ((precision * recall) / (precision + recall))) * 100:.2f}%',
        'Brier score': (1 / sum(n_label)) * brier_score
    }

    return perf_measures


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-type", default = "mBERT", type = str, help = "Model that is trained")
    parser.add_argument("--year", default=2016, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--phase", default="Test", help="The phase of the dataset (Train, Test or Trial)")
    parser.add_argument("--language", default="English", type=str, help="The language of the dataset")
    parser.add_argument("--ont-hops", default=None, type=int, required=False, help="The number of hops in the ontology")
    parser.add_argument("--hops", default=3, type=int,
                        help="The number of hops to use in the rotatory attention mechanism")
    parser.add_argument("--gamma", default=None, type=int, required=False,
                        help="The value of gamma for the LCRRotHopPlusPlus model")
    parser.add_argument("--vm", default=True, type=bool, action=argparse.BooleanOptionalAction,
                        help="Whether to use the visible matrix")
    parser.add_argument("--sp", default=True, type=bool, action=argparse.BooleanOptionalAction,
                        help="Whether to use soft positions")
    parser.add_argument("--ablation", default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help="Run an ablation experiment, this requires all embeddings to exist for a given year.")

    #parser.add_argument("--model", type=str)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model",  type=str, help="Path to a state_dict of the LCRRotHopPlusPlus model")
    group.add_argument("--checkpoint", type=str, help="Path to a checkpoint dir from main_hyperparam.py")

    args = parser.parse_args()

    print("This is the model:", args.model)

    model_type: str = args.model_type
    year: int = args.year
    phase: int = args.phase
    language: str = args.language
    ont_hops: Optional[int] = args.ont_hops
    hops: int = args.hops
    gamma: Optional[int] = args.gamma
    use_vm: bool = args.vm
    use_soft_pos: bool = args.sp
    run_ablation: bool = args.ablation

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    
    input_size = 1024 if model_type == "xlm-roberta-large" else  768
    
    model = LCRRotHopPlusPlus(input_size=input_size, gamma=gamma, hops=hops).to(device)
        #: str= 'data/models/2016_LCR_hops3_dropout0-7000000000000001_acc0-8909574468085106.pt'
        #LCRRotHopPlusPlus(gamma=gamma, hops=hops).to(device)

    if args.model is not None:
        state_dict = torch.load(args.model, map_location=device)
        model.load_state_dict(state_dict)
    elif args.checkpoint is not None:
        state_dict, _ = torch.load(os.path.join(args.checkpoint, "state_dict.pt"), map_location=device)
        model.load_state_dict(state_dict)

    model.eval()

    if not run_ablation:
        dataset = EmbeddingsDataset(model=model_type, year=year, device=device, phase=phase, language=language, ont_hops=ont_hops, use_vm=use_vm,
                                    use_soft_pos=use_soft_pos)
        result = validate_model(model, dataset)

        print("\nResults:")
        for k, v in result.items():
            print(f"  {k}: {v}")

        return

    results_dir = "/content/drive/MyDrive/Thesis_Data/data/results"
    os.makedirs(results_dir, exist_ok=True)
    filename = f"ablation-{year}"
    if gamma is not None:
        filename += f"_gamma-{gamma}"
    csv_writer = CSVWriter(f"{results_dir}/{filename}.csv")

    print('test without knowledge')
    dataset = EmbeddingsDataset(model=model_type, year=year, device=device, phase=phase, ont_hops=ont_hops, use_vm=use_vm,
                                use_soft_pos=use_soft_pos, enable_cache=False)
    result = validate_model(model, dataset, 'LCR-Rot-hop++')
    csv_writer.writerow(result)

    print('test with knowledge injection')
    for ont_hops in range(3):
        for use_vm in [True, False]:
            for use_soft_pos in [True, False]:
                dataset = EmbeddingsDataset(model=model_type, year=year, device=device, phase=phase, ont_hops=ont_hops, use_vm=use_vm,
                                            use_soft_pos=use_soft_pos, enable_cache=False)
                name = f'{ont_hops}-hop LCR-Rot-hop-ont++'

                if not use_vm:
                    name += ' without VM'
                if not use_soft_pos:
                    name += ' without SP'

                result = validate_model(model, dataset, name)
                csv_writer.writerow(result)

    print(f"\nSaved result in {csv_writer.path}")


if __name__ == "__main__":
    main()
