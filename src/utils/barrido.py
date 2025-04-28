import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from random import choices
from tqdm.auto import tqdm
from itertools import product
from typing import List, Dict, Any, Tuple, Optional

from src.config import PATHS
from src.utils.utils_vocab import BasicTokenizer, evaluate 
from src.modelos.bert import BERT


class NNTrainer:

    N_POINTS = 5
    N_EPOCHS = 2 
    LOGGER = list()
    LOGGER_PATH = PATHS["trainer_folder"] / Path('logger_sweep.json')
    TRAINER_PATH = PATHS["trainer_folder"] / Path('trainer.pkl')
    TRAINING_IN_EARNEST = False
    debug = True
    PAD_IDX = 1

    def __init__(self, nombre:str, tokenizer_file:Path) -> None:
        self.nombre = nombre
        special_symbols = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
        simple_tokenizer = lambda tokens_string: tokens_string.strip().split()
        tokenizer = BasicTokenizer.create_using_stoi(simple_tokenizer, special_symbols, tokenizer_file)
        self.tokenizer = tokenizer
        self.crear_puntos()
        self.device = self.get_device()

    def ejecutar_barrido(self) -> None:
        for punto in tqdm(self.puntos, desc='Ejecutando punto...'):
            self.ejecutar_punto(punto)
        print(self.LOGGER)
        with open(self.LOGGER_PATH, 'w') as f:
            json.dump(self.LOGGER, f)
        print('¡Barrido terminado!')

    def ejecutar_punto(
                self, 
                punto:int, 
                save_log:Optional[bool]=True,
                progress_bar:Optional[bool]=False
            ) -> None:
        """
        Ejecuta el entrenamiento del modelo.
        """
        # Crear el modelo
        self.crear_modelo(punto)
        # Entrenar el modelo
        loss, f1, accuracy = self.train_model()
        # Guardar el modelo
        if save_log:
            # Guardar datos en log
            log = dict()
            log['loss'] = loss
            log['f1'] = f1
            log['accuracy'] = accuracy
            self.LOGGER.append(log)
        
    def train_model(self) -> Tuple[float, float, float]:
        # Crear dataloader
        train_dataloader , test_dataloader = self.create_dataloader()
        
        # Entrena el modelo
        # Training loop setup
        num_epochs = self.N_EPOCHS
        total_steps = num_epochs * len(train_dataloader)
        
        # Lists to store losses for plotting
        train_losses = []
        eval_losses = []
        accurracies = []
        f1s = []

        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            self.model.train()
            total_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
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
            eval_loss, acc, f1 = evaluate(test_dataloader, self.model, self.loss_fn_nsp, self.loss_fn_mlm, self.device)
            eval_losses.append(eval_loss)
            accurracies.append(acc)
            f1s.append(f1)
            return eval_losses, accurracies, f1s
    
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

    def get_device(safe=False):
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available() and not safe:
            device = 'mps'
        else:
            device = 'cpu'
        return device

    def obtener_reward(self) -> float:
        mean_reward, std_reward = evaluate_policy(
            self.model, 
            self.model.get_env(), 
            n_eval_episodes=10, 
            deterministic=True,
            return_episode_rewards=False
        )
        return mean_reward

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
        rango_batch_size = [16, 32, 64, 128]
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
    def from_file(env, trainer_path, env_name:Optional[str]='Env') -> 'DQNTrainer':
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
        reward_list = [x['reward'] for x in self.LOGGER]
        best_reward = max(reward_list)
        best_idx = reward_list.index(best_reward)
        parameters = self.LOGGER[best_idx]
        if self.debug:
            print(f'Best reward: {best_reward}')
            print(f'Best learning parameters: {parameters["learning_rate_kwargs"]}')
            print(f'Best policy parameters: {parameters["policy_kwargs"]}')
            print(f'Best model parameters: {parameters["model_kwargs"]}')
        return best_idx
    
    def plot_results(self) -> None:
        log_files = [os.path.join(f"logs_{self.ENV_NAME}", f"_{id}.monitor.csv") for id in range(self.N_ENVS)]
        nrows = np.ceil(self.N_ENVS / 2)
        fig = plt.figure(figsize=(8, 2 * nrows))
        # Create subplots for each log file
        for i, log_file in enumerate(log_files):
            if os.path.isfile(log_file):
                df_results = pd.read_csv(log_file, skiprows=1)
                plt.subplot(int(nrows), 2, i+1, label=log_file)
                df_results['r'].rolling(window=50).mean().plot(title=f"Rewards: Env {i}")
                plt.tight_layout()
        plt.show()
