import argparse
import json
import os
import pickle
import numpy as np
from typing import Optional

import torch
from hyperopt import hp, tpe, fmin, Trials, rand, STATUS_OK
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.lcr_rot_hop_plus_plus import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, train_validation_split
from utils.sent_con_loss import SentConLoss

class HyperOptManager:
    """A class that performs hyperparameter optimization and stores the best states as checkpoints."""


    def __init__(self, model_type:str, year: int, phase: str, language: str, n_epochs_hyper: int, n_epochs_beta: int, contrastive_learning: str, val_ont_hops: Optional[int]):
        self.year = year
        self.phase = phase
        self.language = language
        self.model_type = model_type
        self.n_epochs_hyper = n_epochs_hyper
        self.n_epochs_beta = n_epochs_beta
        self.contrastive_learning = contrastive_learning
        self.val_ont_hops = val_ont_hops

        self.eval_num = 0
        self.best_loss = None
        self.best_hyperparams = None
        self.best_state_dict = None
        self.best_beta = 0.5 if contrastive_learning == "Rep" or contrastive_learning == "Sen" else 0
        self.batch_size = 10 if contrastive_learning == "Rep" else 32
        self.trials = Trials()

        print(torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

        # read checkpoint if exists
        self.__checkpoint_dir = f"/content/drive/MyDrive/Thesis_Data/data/checkpoints/{year}_{phase}_{language}_{self.model_type}_epochs{self.n_epochs_hyper}"

        if contrastive_learning == "Rep":
            self.__checkpoint_dir = self.__checkpoint_dir + "_CL_Rep"
            print("Run model with representation level contrastive learning.")
        elif contrastive_learning == "Sen":
            self.__checkpoint_dir = self.__checkpoint_dir + "_CL_Sen"
            print("Run model with sentiment level contrastive learning.")
        else:
            print("Run model without contrastive learning.")

        if os.path.isdir(self.__checkpoint_dir):
            try:
                self.best_state_dict = torch.load(f"{self.__checkpoint_dir}/state_dict.pt")
                with open(f"{self.__checkpoint_dir}/hyperparams.json", "r") as f:
                    self.best_hyperparams = json.load(f)
                with open(f"{self.__checkpoint_dir}/beta.json", "r") as f:
                    self.best_beta = json.load(f)
                with open(f"{self.__checkpoint_dir}/trials.pkl", "rb") as f:
                    self.trials = pickle.load(f)
                    self.eval_num = len(self.trials)
                with open(f"{self.__checkpoint_dir}/loss.txt", "r") as f:
                    self.best_loss = float(f.read())

                print(f"Resuming from previous checkpoint {self.__checkpoint_dir} with best loss {self.best_loss}")
            except IOError:
                raise ValueError(f"Checkpoint {self.__checkpoint_dir} is incomplete, please remove this directory")
        else:
            print("Starting from scratch")

    def run(self):
        print("running self")
        print("contrastive learning selected") if self.contrastive_learning else print("no contrastive learning selected")
        # TODO: convert to dict for better readability in json file?
        space = [
            hp.choice('learning_rate', [0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.1]),
            hp.quniform('dropout_rate', 0.25, 0.75, 0.1),
            hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),
            hp.choice('weight_decay', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
            hp.choice('lcr_hops', [2, 3, 4, 8])
        ]

        rstate = np.random.default_rng(42)

        best_params = fmin(self.objective, space=space, algo=tpe.suggest, max_evals=5, trials=self.trials,
                           show_progressbar=False, rstate = rstate)
        print("final best: ")
        print(self.best_hyperparams)

        torch.cuda.empty_cache()

        if self.contrastive_learning == "Rep" or self.contrastive_learning == "Sen":

            space = [hp.choice('beta', [0.1,0.2,0.3,0.4])]
            best_params = fmin(self.objective_beta, space=space, algo=tpe.suggest, max_evals=10, trials=self.trials,
                           show_progressbar=False, rstate = rstate)
            
            print(f"\nfinal best beta: {self.best_beta}")

            torch.cuda.empty_cache()


    def objective(self, hyperparams):

        self.eval_num += 1
        learning_rate, dropout_rate, momentum, weight_decay, lcr_hops = hyperparams
        print(f"\n\nEval {self.eval_num} with hyperparams {hyperparams} and beta {self.best_beta}")

        # create training and validation DataLoader
        train_dataset = EmbeddingsDataset(model=self.model_type, year=self.year, phase=self.phase, language=self.language, device=self.device)
        print(f"Using {train_dataset} with {len(train_dataset)} obs for training")
        train_idx, validation_idx = train_validation_split(train_dataset)

        training_subset = Subset(train_dataset, train_idx)

        validation_subset: Subset
        if self.val_ont_hops is not None:
            train_val_dataset = EmbeddingsDataset(model=self.model_type, year=self.year, phase=self.phase, language=self.language, device=self.device,
                                                  ont_hops=self.val_ont_hops)
            validation_subset = Subset(train_val_dataset, validation_idx)
            print(f"Using {train_val_dataset} with {len(validation_subset)} obs for validation")
        else:
            validation_subset = Subset(train_dataset, validation_idx)
            print(f"Using {train_dataset} with {len(validation_subset)} obs for validation")
        training_loader = DataLoader(training_subset, batch_size=self.batch_size, collate_fn=lambda batch: batch)
        validation_loader = DataLoader(validation_subset, collate_fn=lambda batch: batch)

        #Changes input size depending on the model
        input_size = 1024 if self.model_type == "xlm-roberta-large" else  768

        # Train model
        model = LCRRotHopPlusPlus(input_size=input_size,hops=lcr_hops, dropout_prob=dropout_rate).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        best_accuracy: Optional[float] = None
        best_state_dict: Optional[tuple[dict, dict]] = None
        epochs_progress = tqdm(range(self.n_epochs_hyper), unit='epoch')

        for epoch in epochs_progress:
            epoch_progress = tqdm(training_loader, unit='batch', leave=False)
            model.train()

            train_loss = 0.0
            train_n_correct = 0
            train_steps = 0
            train_n = 0

            for i, batch in enumerate(epoch_progress):

                torch.set_default_device(self.device)

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
                loss_cl = SentConLoss().to('cuda')

                if self.contrastive_learning == "Rep":
                    batch_outputs_concat = torch.stack(batch_outputs_concat, dim = 0)
                    loss_cl = loss_cl(batch_outputs_concat, batch_labels)
                    loss_cl = loss_cl / len(batch_outputs_concat)
                else:
                    loss_cl = loss_cl(batch_outputs, batch_labels)
                    loss_cl = loss_cl / len(batch_outputs)

                loss = (1 - self.best_beta) * loss_ce + loss_cl * self.best_beta

                # print(batch_labels)
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
                torch.set_default_device(self.device)

                with torch.no_grad():
                    (left, target, right), label, hops = data[0]

                    if target is not None and len(target) != 0:
                        output, output_concat = model(left, target, right, hops)
                        val_n_correct += (output.argmax(0) == label).type(torch.int).item()
                        val_n += 1

                        loss_ce = criterion(output, label)

                        output = output.unsqueeze(0)
                        output_concat = output_concat.unsqueeze(0)

                        loss_cl = SentConLoss().to('cuda')

                        if self.contrastive_learning == "Rep":
                            loss_cl = loss_cl(output_concat, label)
                            loss_cl = loss_cl / len(output_concat)
                        else:
                            loss_cl = loss_cl(output, label)
                            loss_cl = loss_cl / len(output)

                        loss = (1 - self.best_beta) * loss_ce + loss_cl * self.best_beta

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
                best_state_dict = (model.state_dict(), optimizer.state_dict())

        # we want to maximize accuracy, which is equivalent to minimizing -accuracy
        objective_loss = -best_accuracy
        self.check_best_loss(objective_loss, hyperparams, best_state_dict)
        print("current best: ")
        print(self.best_hyperparams)
        return {
            'loss': loss,
            'status': STATUS_OK,
            'space': hyperparams,
        }

    def check_best_loss(self, loss: float, hyperparams, state_dict: tuple[dict, dict]):
        print("checking best loss")
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_hyperparams = hyperparams
            self.best_state_dict = state_dict

            os.makedirs(self.__checkpoint_dir, exist_ok=True)

            torch.save(state_dict, f"{self.__checkpoint_dir}/state_dict.pt")
            with open(f"{self.__checkpoint_dir}/hyperparams.json", "w") as f:
                json.dump(hyperparams, f)
            with open(f"{self.__checkpoint_dir}/beta.json", "w") as f:
                json.dump(self.best_beta, f)
            with open(f"{self.__checkpoint_dir}/loss.txt", "w") as f:
                f.write(str(self.best_loss))
            print(
                f"Best checkpoint with loss {self.best_loss}, beta {self.best_beta} and hyperparameters {self.best_hyperparams} saved to {self.__checkpoint_dir}")

        with open(f"{self.__checkpoint_dir}/trials.pkl", "wb") as f:
            pickle.dump(self.trials, f)


    def objective_beta(self, beta):

        # Convert tuple to int
        beta = float(beta[0])

        self.eval_num += 1
        learning_rate, dropout_rate, momentum, weight_decay, lcr_hops = self.best_hyperparams
        print(f"\n\nEval {self.eval_num} with hyperparams {self.best_hyperparams} and beta {beta}")

        # create training and validation DataLoader
        train_dataset = EmbeddingsDataset(model=self.model_type, year=self.year, phase=self.phase, language=self.language, device=self.device)
        print(f"Using {train_dataset} with {len(train_dataset)} obs for training")
        train_idx, validation_idx = train_validation_split(train_dataset)

        training_subset = Subset(train_dataset, train_idx)

        validation_subset: Subset
        if self.val_ont_hops is not None:
            train_val_dataset = EmbeddingsDataset(model=self.model_type, year=self.year, phase=self.phase, language=self.language, device=self.device,
                                                  ont_hops=self.val_ont_hops)
            validation_subset = Subset(train_val_dataset, validation_idx)
            print(f"Using {train_val_dataset} with {len(validation_subset)} obs for validation")
        else:
            validation_subset = Subset(train_dataset, validation_idx)
            print(f"Using {train_dataset} with {len(validation_subset)} obs for validation")
        training_loader = DataLoader(training_subset, batch_size=self.batch_size, collate_fn=lambda batch: batch)
        validation_loader = DataLoader(validation_subset, collate_fn=lambda batch: batch)

        # Changes input size depending on the model
        input_size = 1024 if self.model_type == "xlm-roberta-large" else  768

        # Train model
        model = LCRRotHopPlusPlus(input_size=input_size,hops=lcr_hops, dropout_prob=dropout_rate).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        best_accuracy: Optional[float] = None
        best_state_dict: Optional[tuple[dict, dict]] = None
        epochs_progress = tqdm(range(self.n_epochs_beta), unit='epoch')

        for epoch in epochs_progress:
            epoch_progress = tqdm(training_loader, unit='batch', leave=False)
            model.train()

            train_loss = 0.0
            train_n_correct = 0
            train_steps = 0
            train_n = 0

            for i, batch in enumerate(epoch_progress):

                torch.set_default_device(self.device)

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
                loss_cl = SentConLoss().to('cuda')
                
                if self.contrastive_learning == "Rep":
                    batch_outputs_concat = torch.stack(batch_outputs_concat, dim = 0)
                    loss_cl = loss_cl(batch_outputs_concat, batch_labels)
                    loss_cl = loss_cl / len(batch_outputs_concat)
                else:
                    loss_cl = loss_cl(batch_outputs, batch_labels)
                    loss_cl = loss_cl / len(batch_outputs)

                loss = (1 - beta) * loss_ce + loss_cl * beta

                # print(batch_labels)
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
                torch.set_default_device(self.device)

                with torch.no_grad():
                    (left, target, right), label, hops = data[0]

                    if target is not None and len(target) != 0:
                        output, output_concat = model(left, target, right, hops)
                        val_n_correct += (output.argmax(0) == label).type(torch.int).item()
                        val_n += 1

                        loss_ce = criterion(output, label)

                        output = output.unsqueeze(0)
                        output_concat = output_concat.unsqueeze(0)

                        loss_cl = SentConLoss().to('cuda')
                        
                        if self.contrastive_learning == "Rep":
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
                best_state_dict = (model.state_dict(), optimizer.state_dict())

        # we want to maximize accuracy, which is equivalent to minimizing -accuracy
        objective_loss = -best_accuracy
        self.check_best_loss_beta(objective_loss, beta, best_state_dict)
        print(f"current best: {self.best_beta}")

        return {
            'loss': loss,
            'status': STATUS_OK,
            'space': beta,
        }
    
    def check_best_loss_beta(self, loss: float, beta, state_dict: tuple[dict, dict]):
        print(f"\ntested beta: {beta}")
        print(f"\ncurrent accuracy: {loss}")
        print(f"\nbest accuracy: {self.best_loss}")

        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_beta = beta
            self.best_state_dict = state_dict

            os.makedirs(self.__checkpoint_dir, exist_ok=True)

            torch.save(state_dict, f"{self.__checkpoint_dir}/state_dict.pt")
            with open(f"{self.__checkpoint_dir}/beta.json", "w") as f:
                json.dump(beta, f)
            with open(f"{self.__checkpoint_dir}/loss.txt", "w") as f:
                f.write(str(self.best_loss))
            print(
                f"Best checkpoint with loss {self.best_loss}, beta {self.best_beta} and hyperparameters {self.best_hyperparams} saved to {self.__checkpoint_dir}")

        with open(f"{self.__checkpoint_dir}/trials.pkl", "wb") as f:
            pickle.dump(self.trials, f)

def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-type", default="mBERT", type=str, help="the type of model")
    parser.add_argument("--year", default=2016, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--language", default="English", type=str, help="The language of the dataset")
    parser.add_argument("--phase", default="Train", type=str, help="phase of the data")
    parser.add_argument("--val-ont-hops", default=None, type=int, required=False,
                        help="The number of hops to use in the validation phase")
    parser.add_argument("--n-epochs-hyper", default = 20, type = int, help = "Choose number of epochs for hyperparameters. Mostly for debugging")
    parser.add_argument("--n-epochs-beta", default = 20, type = int, help = "Choose number of epochs for beta. Mostly for debugging")
    parser.add_argument("--contrastive-learning", default = None, type = str, 
                        help = "Choose type of contrastive learning for hyperparameter training.")
    
    args = parser.parse_args()
    val_ont_hops: Optional[int] = args.val_ont_hops
    model_type = args.model_type
    year: int = args.year
    language: str = args.language
    phase: str = args.phase
    n_epochs_hyper: int = args.n_epochs_hyper
    n_epochs_beta: int = args.n_epochs_beta
    contrastive_learning: str = args.contrastive_learning

    opt = HyperOptManager(model_type=model_type, year=year, phase=phase, language=language, val_ont_hops=val_ont_hops, 
                          n_epochs_hyper=n_epochs_hyper,n_epochs_beta=n_epochs_beta,contrastive_learning=contrastive_learning)
    opt.run()



if __name__ == "__main__":
    main()
