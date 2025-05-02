import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from random import choices
from tqdm.auto import tqdm
from itertools import product
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split

from src.config import PATHS
from src.modelos.bert import BERT
from src.utils.utils_vocab import BasicTokenizer, BERTDataset, evaluate 


class NNTrainer:

    N_POINTS = 10
    N_EPOCHS = 3
    LOGGER = list()
    LOGGER_PATH = PATHS["trainer_folder"] / Path('logger_sweep.json')
    TRAINER_PATH = PATHS["trainer_folder"] / Path('trainer.pkl')
    TRAINING_IN_EARNEST = False
    debug = True
    PAD_IDX = 1

    def __init__(
                self, 
                nombre: str, 
                tokenizer_file: Path,
                path_dataset: Path
            ) -> None:
        self.nombre = nombre
        special_symbols = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
        simple_tokenizer = lambda tokens_string: tokens_string.strip().split()
        tokenizer = BasicTokenizer.create_using_stoi(simple_tokenizer, special_symbols, tokenizer_file)
        self.tokenizer = tokenizer
        self.crear_puntos()
        self.device = self.get_device()
        self.path_dataset = path_dataset

    def ejecutar_barrido(self) -> None:
        for punto in tqdm(self.puntos, desc='Ejecutando punto...'):
            self.ejecutar_punto(
                    punto, 
                    save_log=not self.TRAINING_IN_EARNEST,
                    progress_bar=self.TRAINING_IN_EARNEST)
        print(self.LOGGER)
        with open(self.LOGGER_PATH, 'w') as f:
            json.dump(self.LOGGER, f)
        print('¡Barrido terminado!')

    def ejecutar_punto(
                self, 
                punto: int, 
                save_log: Optional[bool]=True,
                progress_bar: Optional[bool]=False
            ) -> None:
        """
        Ejecuta el entrenamiento del modelo.
        """
        # Crear el modelo
        self.crear_modelo(punto)
        batch_size = punto[7]
        # Entrenar el modelo
        try:
            results = self.train_model(batch_size, progress_bar=progress_bar)
            loss = results['test']['loss'][-1]
            f1 = results['test']['f1'][-1]
            accurracy = results['test']['accuracy'][-1]
            msg = 'No errors'
        except Exception as e:
            loss, f1, accurracy = np.infty, np.infty, np.infty
            msg = f'Error: {e}'
            results = None
        # Guardar el modelo
        if save_log:
            # Guardar datos en log
            log = dict()
            log['loss'] = loss
            log['accurracy'] = accurracy
            log['f1'] = f1
            log['hiperparametros'] = punto
            log['message'] = msg
            self.LOGGER.append(log)
        else:
            return results
        
    def train_model(
                self, 
                batch_size: int, 
                progress_bar: Optional[bool]=False
            ) -> Tuple[float, float, float]:
        # Crear dataloader
        train_dataloader , test_dataloader = self.create_dataloader(batch_size)
        
        # Entrena el modelo
        # Training loop setup
        num_epochs = self.N_EPOCHS
        total_steps = num_epochs * len(train_dataloader)
        
        # Lists to store losses for plotting
        train_losses = []
        eval_losses = []
        train_accurracies = []
        eval_accurracies = []
        train_f1s = []
        eval_f1s = []

        progress2 = tqdm(range(num_epochs), desc="Training Epochs", leave=not progress_bar)
        for epoch in progress2:
            self.model.train()
            total_loss = 0

            progress1 = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False, disable=None)
            for step, batch in enumerate(progress1):
                bert_inputs, bert_labels, segment_labels, is_nexts = [b.to(self.device) for b in batch]

                self.optimizer.zero_grad()
                next_sentence_prediction, masked_language = self.model(bert_inputs, segment_labels)

                next_loss = self.loss_fn_nsp(next_sentence_prediction, is_nexts)
                mask_loss = self.loss_fn_mlm(masked_language.view(-1, masked_language.size(-1)), bert_labels.view(-1))

                loss = next_loss + mask_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()  # Update the learning rate

                #total_loss += loss.item()

                if torch.isnan(loss):

                    raise Exception(f'{loss.item()=}---{next_sentence_prediction=}\n{masked_language=}\n{next_loss=}\n{mask_loss=}\n{total_loss=}\n {bert_inputs=}\n{bert_labels=}\n{segment_labels=}\n{is_nexts=}') 
                else:
                    total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dataloader) + 1
            assert(not np.isnan(avg_train_loss)), f'{loss.item()=}---{total_loss=}\n {bert_inputs=}\n{bert_labels=}\n{segment_labels=}\n{is_nexts=}'
            train_losses.append(avg_train_loss)
            #print(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")

            # Evaluation after each epoch
            _, train_acc, train_f1 = evaluate(train_dataloader, self.model, self.loss_fn_nsp, self.loss_fn_mlm, self.device)
            train_accurracies.append(train_acc)
            train_f1s.append(train_f1)
            
            eval_loss, acc, f1 = evaluate(test_dataloader, self.model, self.loss_fn_nsp, self.loss_fn_mlm, self.device)
            eval_losses.append(eval_loss)
            eval_accurracies.append(acc)
            eval_f1s.append(f1)
        
        results = dict()
        results['training'] = {
            'loss': train_losses,
            'accurracy': train_acc,
            'f1': train_f1s
        }
        results['test'] = {
            'loss': eval_losses,
            'accurracy': eval_accurracies,
            'f1': eval_f1s
        }

        return results
    
    def crear_modelo(self, punto:List[Tuple[any]]) -> None:
        """
        Crea el modelo de entrenamiento.
        """
        # Define parameters
        vocab_size = self.tokenizer.get_vocab_size()  # Replace VOCAB_SIZE with your vocabulary size
        d_model = punto[0]  # Replace EMBEDDING_DIM with your embedding dimension
        n_layers = punto[1]  # Number of Transformer layers
        initial_heads = punto[2]  # Number of attention heads
        # Ensure the number of heads is a factor of the embedding dimension
        heads = initial_heads - d_model % initial_heads

        dropout = punto[3]  # Dropout rate

        # Create an instance of the BERT model
        self.model = BERT(vocab_size, d_model, n_layers, heads, dropout)
        self.model.to(self.device)
        # Define the optimizer
        self.optimizer = self.crear_optimizer(punto[4], punto[5])
        # Define the learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=punto[6])
        # Define the loss functions
        self.loss_fn_mlm = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)
        self.loss_fn_nsp = torch.nn.CrossEntropyLoss()

    def create_dataloader(self, batch_size: int) -> Tuple[DataLoader]:
        df = pd.read_csv(self.path_dataset)
        # Define features (X) and target (y)
        X = df[['bert_input', 'segment_label']]  # Replace with your feature columns
        y = df[['bert_label', 'relation']]  # Replace with your target column

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        df_train = pd.DataFrame({
            'BERT Input': X_train['bert_input'],
            'Segment Label': X_train['segment_label'],
            'BERT Label': y_train['bert_label'],
            'Is Next': y_train['relation']
        })

        df_test = pd.DataFrame({
            'BERT Input': X_test['bert_input'],
            'Segment Label': X_test['segment_label'],
            'BERT Label': y_test['bert_label'],
            'Is Next': y_test['relation']
        })

        train_dataset = BERTDataset(df_train)
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=self.collate_batch
        )

        test_dataset = BERTDataset(df_test)
        test_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self.collate_batch
        )

        return train_dataloader, test_dataloader

    def collate_batch(self, batch):
        bert_inputs_batch, bert_labels_batch, segment_labels_batch, is_nexts_batch = [], [], [], []

        for bert_input, bert_label, segment_label, is_next in batch:
            # Convert each sequence to a tensor and append to the respective list
            bert_inputs_batch.append(torch.tensor(bert_input, dtype=torch.long))
            bert_labels_batch.append(torch.tensor(bert_label, dtype=torch.long))
            segment_labels_batch.append(torch.tensor(segment_label, dtype=torch.long))
            is_nexts_batch.append(is_next)

        # Pad the sequences in the batch
        bert_inputs_final = pad_sequence(bert_inputs_batch, padding_value=self.PAD_IDX, batch_first=False)
        bert_labels_final = pad_sequence(bert_labels_batch, padding_value=self.PAD_IDX, batch_first=False)
        segment_labels_final = pad_sequence(segment_labels_batch, padding_value=self.PAD_IDX, batch_first=False)
        is_nexts_batch = torch.tensor(is_nexts_batch, dtype=torch.long)

        return bert_inputs_final.to(self.device), bert_labels_final.to(self.device), segment_labels_final.to(self.device), is_nexts_batch.to(self.device)

    def get_device(safe=False):
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available() and not safe:
            device = 'mps'
        else:
            device = 'cpu'
        return device

    def generar_parametros_entrenamiento(
                self, 
                punto:List[Any]
            ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        learning_rate_initial = punto[0]
        learning_rate_final =  punto[1]
        net_arch=punto[2]
        optimizer_class=punto[3]
        buffer_size=punto[4]
        target_update_interval=punto[5]
        gamma=punto[6]
        exploration_fraction = punto[7]
        exploration_initial_eps= punto[8]
        exploration_final_eps= punto[9]
        batch_size = punto[10]
        learning_rate_kwargs = {
            "initial_learning_rate": learning_rate_initial,
            "final_learning_rate": learning_rate_final,
        }
        # print(f'{learning_rate_kwargs=}')
        policy_kwargs = {           
            "net_arch": net_arch,
            "optimizer_class": optimizer_class,
        }
        # print(f'{policy_kwargs=}')
        model_kwargs = {
            "buffer_size": buffer_size,
            "target_update_interval": target_update_interval,
            "gamma": gamma,
            "exploration_fraction": exploration_fraction,
            "exploration_initial_eps": exploration_initial_eps,
            "exploration_final_eps": exploration_final_eps,
            "batch_size": batch_size,
        }
        # print(f'{model_kwargs=}')
        return learning_rate_kwargs, policy_kwargs, model_kwargs
    
    def crear_optimizer(self, optimizer:str, learning_rate:float) -> torch.optim:
        if optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise Exception(f'Optimizer {optimizer} not accepted. Choose from {self.accepted_optimizers}')

    def crear_puntos(self) -> None:
        rango_embedding_dim = [4, 8, 16, 32]
        rango_net_layers = [4, 6, 8, 10]
        rango_net_heads = [2, 4, 6, 8]
        rango_dropout = [0.1, 0.2, 0.3, 0.4]
        rango_optimizer_class = ['adam', 'sgd']
        rango_learning_rate_initial = [0.01, 0.001, 0.0001, 0.00001]
        rango_gamma = [0.1, 0.2, 0.3, 0.4]
        rango_batch_size = [1, 4, 16, 32, 64]
        opciones = product(
            rango_embedding_dim,
            rango_net_layers,
            rango_net_heads,
            rango_dropout,
            rango_optimizer_class,
            rango_learning_rate_initial,
            rango_gamma,
            rango_batch_size
        )
        puntos = choices(
            list(opciones), 
            k=self.N_POINTS
        )
        self.puntos = puntos

    def save(self) -> None:
        dict_self = {
            'puntos':self.puntos,
            'atributos': {
                'ENV_NAME':self.ENV_NAME,
                'N_ENVS':self.N_ENVS,
                'N_POINTS':self.N_POINTS,
                'TIME_STEPS':self.TIME_STEPS
            }
        }
        with open(self.TRAINER_PATH, 'wb') as f:
            pickle.dump(dict_self, f)
        print(f'Self guardado con éxito en {self.TRAINER_PATH}')

    @staticmethod
    def from_file(env, trainer_path, env_name:Optional[str]='Env') -> 'NNTrainer':
        # Load saved pickle
        with open(trainer_path, 'rb') as f:
            dict_self = pickle.load(f)
        # Create trainer
        trainer = DQNTrainer(env, env_name)
        # Update logger
        with open(trainer.LOGGER_PATH, 'r') as f:
            trainer.LOGGER = json.load(f)
        # Update attributes
        atributos = dict_self['atributos']
        for atributo, value in atributos.items():
            setattr(trainer, atributo, value)
        # Update puntos
        trainer.puntos = dict_self['puntos']
        print(f'¡Trainer creado desde {trainer_path} con éxito!')
        return trainer

    def retornar_mejor_combinacion_idx(self):
        f1_list = [x['f1'] for x in self.LOGGER]
        best_f1 = max(f1_list)
        best_idx = f1_list.index(best_f1)
        punto = self.LOGGER[best_idx]['hiperparametros']
        parameters = self.punto_a_dict(punto)
        if self.debug:
            print(f'Best f1: {best_f1}')
            for parameter, value in parameters.items():
                print(f'Best {parameter}: {value}')
        return best_idx

    def punto_a_dict(self, punto: Tuple[any]) -> Dict[str, any]:
        hiperparametros = dict()
        hiperparametros['embedding_dim'] = punto[0]
        hiperparametros['net_layers'] = punto[1]
        hiperparametros['net_heads'] = punto[2]
        hiperparametros['dropout'] = punto[3]
        hiperparametros['optimizer_class'] = punto[4]
        hiperparametros['learning_rate_initial'] = punto[5]
        hiperparametros['gamma'] = punto[6]
        hiperparametros['batch_size'] = punto[7]
        return hiperparametros

    def plot_results(self, XXXXX) -> None:

        batch_size = ...
        # Crear dataloader
        train_dataloader , test_dataloader = self.create_dataloader(batch_size)

        fig,axes = plt.subplots(
            1,3,
            figsize=(9,3),
            tight_layout=True
        )
        sns.lineplot(ax = axes[0], x=range(1, self.N_EPOCHS + 1), y=train_losses, label='Training Loss')
        sns.lineplot(ax = axes[0], x=range(1, self.N_EPOCHS + 1), y=eval_losses, label='Evaluation Loss')
        sns.lineplot(ax = axes[1], x=range(1, self.N_EPOCHS + 1), y=accurracies, label='Accurracy')
        sns.lineplot(ax = axes[2], x=range(1, self.N_EPOCHS + 1), y=f1s, label='F1 score')

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accurracy')
        axes[2].set_ylabel('F1 score')
        axes[0].set_title('Training and Evaluation Loss')
        axes[1].set_title('Accurracy')
        axes[2].set_title('F1 score')
        plt.legend()
        plt.show()