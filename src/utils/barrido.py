import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from random import choices
from tqdm.auto import tqdm
from itertools import product
from typing import List, Dict, Any, Tuple, Optional
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

class DQNTrainer:

    N_ENVS = 1
    N_POINTS = 5
    TIME_STEPS = 10_000   
    LOGGER = list()
    LOGGER_PATH = Path('logger_sweep.json')
    TRAINER_PATH = Path('trainer.pkl')
    accepted_optimizers = ['adam', 'sgd']
    TRAINING_IN_EARNEST = False
    debug = True

    def __init__(self, env, env_name:Optional[str]='Env') -> None:
        self.env = env
        self.ENV_NAME = env_name
        self.crear_puntos()

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
        # Generar los parámetros de entrenamiento
        learning_rate_kwargs, policy_kwargs, model_kwargs = self.generar_parametros_entrenamiento(punto)
        # Crear el modelo
        self.crear_modelo(learning_rate_kwargs, policy_kwargs, model_kwargs)
        # Entrenar el modelo
        if self.TRAINING_IN_EARNEST: progress_bar = True
        self.model.learn(
            total_timesteps=self.TIME_STEPS, 
            reset_num_timesteps=False, 
            progress_bar=progress_bar
        )
        if save_log:
            # Guardar datos en log
            log = dict()
            log['learning_rate_kwargs'] = learning_rate_kwargs
            log['policy_kwargs'] = policy_kwargs
            log['model_kwargs'] = model_kwargs
            reward = self.obtener_reward()
            log['reward'] = reward
            self.LOGGER.append(log)
    
    def make_env(self, ENV_NAME, id:int=0) -> gym.Env:
        # Creamos el entorno de mercado
        def _init():
            if self.TRAINING_IN_EARNEST:
                log_file = os.path.join(f"logs_{ENV_NAME}", f"_{id}")
                env = Monitor(deepcopy(self.env), log_file, allow_early_resets=True)
            else:
                env = deepcopy(self.env)
            return env
        return _init

    def crear_modelo(self, learning_rate_kwargs, policy_kwargs:Dict[str, Any], model_kwargs:Dict[str, Any]) -> None:
        """
        Crea el modelo de entrenamiento.
        """
        # Funcion asocida a la politica de exploracion con base en el proceso de entrenamiento
        def learning_rate_fn(process_remaining):
            initial = learning_rate_kwargs['initial_learning_rate']
            final =  learning_rate_kwargs['final_learning_rate']
            return final + (initial - final) * process_remaining
        
        # Crear el entorno de entrenamiento
        train_env = DummyVecEnv([self.make_env(self.ENV_NAME, i) for i in range(self.N_ENVS)])        
        # Actualizar parametros policy con el optmizer
        policy_kwargs_ = policy_kwargs.copy()
        policy_kwargs_['optimizer_class'] = self.crear_optimizer(policy_kwargs['optimizer_class'])
        # Asignar parametros modelo
        model_kwargs_ = dict(
            policy="MlpPolicy",
            seed=42,
            policy_kwargs=policy_kwargs_,
            learning_rate=learning_rate_fn,
            stats_window_size=50,
            verbose=0
        )        
        model_kwargs_.update(model_kwargs)
        # Instanciar el agente de entrenamiento
        self.model = DQN(env=train_env, **model_kwargs_)

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
    
    def crear_optimizer(self, optimizer:str) -> torch.optim:
        if optimizer == "adam":
            return torch.optim.Adam
        elif optimizer == "sgd":
            return torch.optim.SGD
        else:
            raise Exception(f'Optimizer {optimizer} not accepted. Choose from {self.accepted_optimizers}')

    def crear_puntos(self) -> None:
        rango_learning_rate_initial = [0.01, 0.001, 0.0001]
        rango_learning_rate_final = [0.01 * x for x in rango_learning_rate_initial]
        rango_net_arch = [[64, 32], [128, 64], [256, 128]]
        rango_optimizer_class = ['adam', 'sgd']
        rango_buffer_size = [1000, 2000, 5000]
        rango_target_update_interval = [100, 500, 1000]
        rango_gamma = [0.9, 0.95, 0.99]
        rango_exploration_fraction = [0.5, 0.6, 0.7]
        rango_exploration_initial_eps = [0.8, 0.9, 1.0]
        rango_exploration_final_eps = [0.01, 0.1, 0.2]
        rango_batch_size = [32, 64, 128]
        opciones = product(
            rango_learning_rate_initial,
            rango_learning_rate_final,
            rango_net_arch,
            rango_optimizer_class,
            rango_buffer_size,
            rango_target_update_interval,
            rango_gamma,
            rango_exploration_fraction,
            rango_exploration_initial_eps,
            rango_exploration_final_eps,
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
