import argparse
import os
from typing import Optional

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.lcr_rot_hop_plus_plus import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, train_validation_split
from utils.con_loss import ConLoss


def stringify_float(value: float):
    return str(value).replace('.', '-')


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-type", default = "mBERT", type = str, help = "Model that is trained")
    parser.add_argument("--year", default=2016, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--language", default="English", type=str, help="The language of the dataset")
    parser.add_argument("--phase", default="Train", help="The phase of the dataset (Train or Test)")
    parser.add_argument("--hops", default=3, type=int,
                        help="The number of hops to use in the rotatory attention mechanism")
    parser.add_argument("--ont-hops", default=None, type=int, required=False,
                        help="The number of hops in the ontology to use")
    parser.add_argument("--val-ont-hops", default=None, type=int, required=False,
                        help="The number of hops to use in the validation phase, this option overrides the --ont-hops option.")
    parser.add_argument("--learning",default=0.01,type=float,help="learning rate from hyperparamter tuning")
    parser.add_argument("--dropout", default=0.30000000000000004, type=float,help="dropout rate from hyperparameter tuning")
    parser.add_argument("--momentum",default=0.99,type=float,help="momentum from hyperparameter tuning")
    parser.add_argument("--weight-decay",default=0.0001,type=float,help="weight decay from hyperparameter tuning")
    parser.add_argument("--n-epochs", default = 100, type = int, help = "Choose number of epochs. Mostly for debugging")
    parser.add_argument("--beta", default = 0, type = float, 
                        help = "Hyperparameter for contrastive learning. 0: No contrastive, 1: Only contrastive")
    parser.add_argument("--contrastive-learning", default = None, type = str, 
                        help = "Choose type of contrastive learning for hyperparameter training.")
    
    args = parser.parse_args()

    model_type: str = args.model_type
    year: int = args.year
    phase: str = args.phase
    language: str = args.language

    lcr_hops: int = args.hops
    ont_hops: Optional[int] = args.ont_hops
    val_ont_hops: Optional[int] = args.val_ont_hops
    dropout_rate: float = args.dropout
    learning_rate: float = args.learning
    momentum: float = args.momentum
    weight_decay: float = args.weight_decay
    n_epochs: int = args.n_epochs
    beta: float = args.beta 
    contrastive_learning: str = args.contrastive_learning
    

    batch_size = 10 if contrastive_learning == "Rep" else 32

    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    # create training anf validation DataLoader
    train_dataset = EmbeddingsDataset(model = model_type, year=year, device=device, phase=phase, language=language, ont_hops=ont_hops)
    print(f"Using {train_dataset} with {len(train_dataset)} obs for training")
    train_idx, validation_idx = train_validation_split(train_dataset)

    training_subset = Subset(train_dataset, train_idx)

    if val_ont_hops is not None:
        train_val_dataset = EmbeddingsDataset(model = model_type, year=year, device=device, phase=phase, ont_hops=val_ont_hops)
        validation_subset = Subset(train_val_dataset, validation_idx)
        print(f"Using {train_val_dataset} with {len(validation_subset)} obs for validation")
    else:
        validation_subset = Subset(train_dataset, validation_idx)
        print(f"Using {train_dataset} with {len(validation_subset)} obs for validation")

    training_loader = DataLoader(training_subset, batch_size=batch_size, collate_fn=lambda batch: batch)
    validation_loader = DataLoader(validation_subset, collate_fn=lambda batch: batch)

    input_size = 1024 if model_type == "xlm-roberta-large" else  768

    # Train model
    model = LCRRotHopPlusPlus(input_size=input_size, hops=lcr_hops, dropout_prob=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    best_accuracy: Optional[float] = None
    best_state_dict: Optional[dict] = None
    epochs_progress = tqdm(range(n_epochs), unit='epoch')

    try:
        for epoch in epochs_progress:
            epoch_progress = tqdm(training_loader, unit='batch', leave=False)
            model.train()

            train_loss = 0.0
            train_n_correct = 0
            train_steps = 0
            train_n = 0

            for i, batch in enumerate(epoch_progress):
                torch.set_default_device(device)

                #This provides a failsafe if target is empty
                batch_outputs_concat = []
                batch_outputs =[]
                batch_labels=[]

                for (left, target, right), label, hops in batch:
                    if target is not None and len(target) != 0:
                        output, output_concat = model(left, target, right, hops)

                        batch_outputs_concat.append(output_concat)
                        batch_outputs.append(output)
                        batch_labels.append(label.item())
                    else:
                        print(f"\n[Empty target in instance in batch {i}. Instance skipped]")

                batch_outputs = torch.stack(batch_outputs, dim = 0)
                batch_labels = torch.tensor(batch_labels)

                loss_ce: torch.Tensor = criterion(batch_outputs, batch_labels)
                loss_cl = ConLoss().to('cuda')

                if contrastive_learning == "Rep":
                    batch_outputs_concat = torch.stack(batch_outputs_concat, dim = 0)
                    loss_cl = loss_cl(batch_outputs_concat, batch_labels)
                    loss_cl = loss_cl / len(batch_outputs_concat)
                else:
                    loss_cl = loss_cl(batch_outputs, batch_labels)
                    loss_cl = loss_cl / len(batch_outputs)

                loss = (1 - beta) * loss_ce + loss_cl * beta 

                train_loss += loss.item()
                train_steps += 1
                train_n_correct += (batch_outputs.argmax(1) == batch_labels).type(torch.int).sum().item()
                train_n += len(batch)

                epoch_progress.set_description(
                    f"Train Loss: {train_loss / train_steps:.3f}, Train Acc.: {train_n_correct / train_n:.3f}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                torch.set_default_device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

            # Validation loss
            epoch_progress = tqdm(validation_loader, unit='obs', leave=False)
            model.eval()

            val_loss = 0.0
            val_steps = 0
            val_n = 0
            val_n_correct = 0
            for i, data in enumerate(epoch_progress):
                torch.set_default_device(device)

                with torch.no_grad():
                    (left, target, right), label, hops = data[0]

                    if target is not None and len(target) != 0:
                        output, output_concat = model(left, target, right, hops)
                        val_n_correct += (output.argmax(0) == label).type(torch.int).item()
                        val_n += 1

                        loss_ce: torch.Tensor = criterion(output, label)

                        output = output.unsqueeze(0)
                        output_concat = output_concat.unsqueeze(0)

                        loss_cl = ConLoss().to('cuda')
                    
                        if contrastive_learning == "Rep":
                            loss_cl = loss_cl(output_concat, label)
                            loss_cl = loss_cl / len(output_concat)
                        else:
                            loss_cl = loss_cl(output, label)
                            loss_cl = loss_cl / len(output)
                        
                        loss = (1 - beta) * loss_ce + loss_cl * beta 

                        val_loss += loss.item()
                        val_steps += 1

                        epoch_progress.set_description(
                            f"Test Loss: {val_loss / val_steps:.3f}, Test Acc.: {val_n_correct / val_n:.3f}")
                    else:
                        print(f"\n[Invalid instance in validation batch {i}. Instance skipped]")

                torch.set_default_device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

            validation_accuracy = val_n_correct / val_n

            if best_accuracy is None or validation_accuracy > best_accuracy:
                epochs_progress.set_description(f"Best Test Acc.: {validation_accuracy:.3f}")
                best_accuracy = validation_accuracy
                best_state_dict = model.state_dict()
    except KeyboardInterrupt:
        print("Interrupted training procedure, saving best model...")

    if best_state_dict is not None:
        models_dir = os.path.join("/content/drive/MyDrive/data", "models")
        os.makedirs(models_dir, exist_ok=True)
        model_name = f"{year}_{language}_LCR_hops{lcr_hops}_dropout{stringify_float(dropout_rate)}_acc{stringify_float(best_accuracy)}_{model_type}"

        if contrastive_learning == "Sen":
            model_name = model_name + "_CL_Sen.pt"
        elif contrastive_learning == "Rep":
             model_name = model_name + "_CL_Rep.pt"
        else:
            model_name = model_name + ".pt"
        
        model_path = os.path.join(models_dir,model_name)
        with open(model_path, "wb") as f:
            torch.save(best_state_dict, f)
            print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
