#!/usr/bin/env python
# Cosmic Market Oracle - Hyperparameter Tuning with Optuna

"""
Hyperparameter tuning module for the Cosmic Market Oracle.

This module implements Optuna-based hyperparameter optimization for deep learning models,
with multi-objective optimization, pruning strategies, and visualization capabilities.
"""

import os
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
import mlflow
import matplotlib.pyplot as plt

from src.utils.logger import setup_logger
from src.models.model_factory import ModelFactory
from src.utils.config import Config

# Configure logging
logger = setup_logger("hyperparameter_tuning")

class OptunaHyperparameterTuner:
    """
    Optuna-based hyperparameter tuning for deep learning models.
    
    This class provides a comprehensive framework for optimizing model hyperparameters
    using Optuna, with support for multi-objective optimization, pruning strategies,
    and visualization of results.
    """
    
    def __init__(
        self,
        config_path: str,
        data_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        study_name: str = "cosmic_market_oracle_optimization",
        storage: Optional[str] = None,
        direction: Union[str, List[str]] = "maximize",
        pruner: Optional[optuna.pruners.BasePruner] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            config_path: Path to the configuration file
            data_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data (optional)
            n_trials: Number of optimization trials
            timeout: Timeout for optimization in seconds (optional)
            study_name: Name of the Optuna study
            storage: Optuna storage URL (optional)
            direction: Optimization direction ("maximize" or "minimize", or list for multi-objective)
            pruner: Optuna pruner (optional)
            sampler: Optuna sampler (optional)
        """
        self.config = Config(config_path)
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        
        # Set default pruner if not provided
        if pruner is None:
            self.pruner = MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=1
            )
        else:
            self.pruner = pruner
            
        # Set default sampler if not provided
        if sampler is None:
            self.sampler = optuna.samplers.TPESampler(seed=42)
        else:
            self.sampler = sampler
            
        # Create Optuna study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            pruner=self.pruner,
            sampler=self.sampler,
            load_if_exists=True
        )
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(f"hyperparameter_tuning_{self.study_name}")
        
        # Initialize best parameters and model
        self.best_params = None
        self.best_model = None
        self.best_score = None
        
    def define_model_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for the model.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        # Model architecture parameters
        params = {
            # Model type
            "model_type": trial.suggest_categorical(
                "model_type", ["lstm", "tcn", "transformer", "gnn", "hybrid"]
            ),
            
            # General parameters
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            
            # Sequence parameters
            "sequence_length": trial.suggest_categorical(
                "sequence_length", [30, 60, 90, 120, 180, 360]
            ),
            
            # Embedding parameters
            "embedding_dim": trial.suggest_int("embedding_dim", 16, 256, step=16),
            
            # Optimizer
            "optimizer": trial.suggest_categorical(
                "optimizer", ["adam", "adamw", "sgd", "rmsprop"]
            ),
            
            # Scheduler
            "scheduler": trial.suggest_categorical(
                "scheduler", ["cosine", "plateau", "step", "none"]
            ),
        }
        
        # Model-specific parameters
        if params["model_type"] == "lstm":
            params.update({
                "hidden_size": trial.suggest_int("hidden_size", 32, 512, step=32),
                "num_layers": trial.suggest_int("num_layers", 1, 5),
                "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
                "attention_heads": trial.suggest_int("attention_heads", 1, 8),
            })
        elif params["model_type"] == "tcn":
            params.update({
                "num_channels": trial.suggest_int("num_channels", 32, 256, step=32),
                "kernel_size": trial.suggest_int("kernel_size", 2, 8),
                "num_levels": trial.suggest_int("num_levels", 2, 8),
            })
        elif params["model_type"] == "transformer":
            params.update({
                "num_heads": trial.suggest_int("num_heads", 2, 16, step=2),
                "num_layers": trial.suggest_int("num_layers", 1, 6),
                "dim_feedforward": trial.suggest_int("dim_feedforward", 128, 1024, step=128),
            })
        elif params["model_type"] == "gnn":
            params.update({
                "gnn_layers": trial.suggest_int("gnn_layers", 1, 5),
                "gnn_hidden_dim": trial.suggest_int("gnn_hidden_dim", 32, 256, step=32),
                "aggregation": trial.suggest_categorical(
                    "aggregation", ["sum", "mean", "max"]
                ),
            })
        elif params["model_type"] == "hybrid":
            params.update({
                "cnn_filters": trial.suggest_int("cnn_filters", 16, 256, step=16),
                "cnn_kernel_size": trial.suggest_int("cnn_kernel_size", 2, 8),
                "transformer_heads": trial.suggest_int("transformer_heads", 2, 8),
                "transformer_layers": trial.suggest_int("transformer_layers", 1, 4),
            })
            
        return params
    
    def objective(self, trial: optuna.Trial) -> Union[float, List[float]]:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation metric(s) to optimize
        """
        # Get hyperparameters for this trial
        params = self.define_model_parameters(trial)
        
        # Log parameters to MLflow
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            
            try:
                # Create model
                model_factory = ModelFactory(params)
                model = model_factory.create_model()
                
                # Configure optimizer
                if params["optimizer"] == "adam":
                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=params["learning_rate"],
                        weight_decay=params["weight_decay"]
                    )
                elif params["optimizer"] == "adamw":
                    optimizer = optim.AdamW(
                        model.parameters(),
                        lr=params["learning_rate"],
                        weight_decay=params["weight_decay"]
                    )
                elif params["optimizer"] == "sgd":
                    optimizer = optim.SGD(
                        model.parameters(),
                        lr=params["learning_rate"],
                        momentum=0.9,
                        weight_decay=params["weight_decay"]
                    )
                elif params["optimizer"] == "rmsprop":
                    optimizer = optim.RMSprop(
                        model.parameters(),
                        lr=params["learning_rate"],
                        weight_decay=params["weight_decay"]
                    )
                
                # Configure scheduler
                if params["scheduler"] == "cosine":
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=10
                    )
                elif params["scheduler"] == "plateau":
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.5, patience=5
                    )
                elif params["scheduler"] == "step":
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer, step_size=5, gamma=0.5
                    )
                else:
                    scheduler = None
                
                # Configure loss function
                criterion = nn.MSELoss()
                
                # Train for a fixed number of epochs
                num_epochs = 20
                best_val_loss = float("inf")
                best_val_accuracy = 0.0
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                for epoch in range(num_epochs):
                    # Training
                    model.train()
                    train_loss = 0.0
                    for batch_idx, (data, target) in enumerate(self.data_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                    
                    train_loss /= len(self.data_loader)
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for data, target in self.val_loader:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            val_loss += criterion(output, target).item()
                            
                            # For directional accuracy
                            pred_direction = (output[:, 0] > 0).float()
                            true_direction = (target[:, 0] > 0).float()
                            correct += (pred_direction == true_direction).sum().item()
                            total += target.size(0)
                    
                    val_loss /= len(self.val_loader)
                    val_accuracy = correct / total
                    
                    # Update scheduler if needed
                    if scheduler is not None:
                        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(val_loss)
                        else:
                            scheduler.step()
                    
                    # Log metrics
                    mlflow.log_metrics({
                        f"train_loss_epoch_{epoch}": train_loss,
                        f"val_loss_epoch_{epoch}": val_loss,
                        f"val_accuracy_epoch_{epoch}": val_accuracy
                    }, step=epoch)
                    
                    # Report intermediate values for pruning
                    trial.report(val_loss, epoch)
                    
                    # Handle pruning
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    
                    # Track best validation metrics
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                
                # Log final metrics
                mlflow.log_metrics({
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy
                })
                
                # For multi-objective optimization
                if isinstance(self.direction, list):
                    return [best_val_accuracy, -best_val_loss]  # Maximize accuracy, minimize loss
                else:
                    return best_val_accuracy  # Default: maximize accuracy
                
            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {str(e)}")
                mlflow.log_param("error", str(e))
                raise optuna.exceptions.TrialPruned()
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the hyperparameter optimization process.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        start_time = time.time()
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best trial and parameters
        best_trial = self.study.best_trial
        self.best_params = best_trial.params
        self.best_score = best_trial.value
        
        # Log best parameters and score
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best score: {self.best_score}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Log optimization time
        optimization_time = time.time() - start_time
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        
        # Create and save visualizations
        self._save_visualizations()
        
        # Train best model
        self._train_best_model()
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "best_trial": best_trial.number,
            "optimization_time": optimization_time,
            "n_trials": len(self.study.trials),
            "completed_trials": len(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
            "pruned_trials": len(self.study.get_trials(states=[optuna.trial.TrialState.PRUNED])),
        }
    
    def _save_visualizations(self) -> None:
        """Save optimization visualizations."""
        # Create output directory
        output_dir = Path("results/hyperparameter_tuning")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameter importance plot
        param_importance_fig = plot_param_importances(self.study)
        param_importance_fig.write_image(str(output_dir / "param_importance.png"))
        
        # Optimization history plot
        optimization_history_fig = plot_optimization_history(self.study)
        optimization_history_fig.write_image(str(output_dir / "optimization_history.png"))
        
        # Save study as JSON
        with open(output_dir / "study_results.json", "w") as f:
            json.dump({
                "best_params": self.best_params,
                "best_score": self.best_score,
                "study_name": self.study_name,
                "direction": self.direction,
                "n_trials": self.n_trials,
                "completed_trials": len(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
                "pruned_trials": len(self.study.get_trials(states=[optuna.trial.TrialState.PRUNED])),
            }, f, indent=2)
    
    def _train_best_model(self) -> None:
        """Train the best model with the optimized hyperparameters."""
        logger.info("Training best model with optimized hyperparameters")
        
        with mlflow.start_run(run_name="best_model_training"):
            # Log best parameters
            mlflow.log_params(self.best_params)
            
            # Create model with best parameters
            model_factory = ModelFactory(self.best_params)
            self.best_model = model_factory.create_model()
            
            # Configure optimizer
            if self.best_params["optimizer"] == "adam":
                optimizer = optim.Adam(
                    self.best_model.parameters(),
                    lr=self.best_params["learning_rate"],
                    weight_decay=self.best_params["weight_decay"]
                )
            elif self.best_params["optimizer"] == "adamw":
                optimizer = optim.AdamW(
                    self.best_model.parameters(),
                    lr=self.best_params["learning_rate"],
                    weight_decay=self.best_params["weight_decay"]
                )
            elif self.best_params["optimizer"] == "sgd":
                optimizer = optim.SGD(
                    self.best_model.parameters(),
                    lr=self.best_params["learning_rate"],
                    momentum=0.9,
                    weight_decay=self.best_params["weight_decay"]
                )
            elif self.best_params["optimizer"] == "rmsprop":
                optimizer = optim.RMSprop(
                    self.best_model.parameters(),
                    lr=self.best_params["learning_rate"],
                    weight_decay=self.best_params["weight_decay"]
                )
            
            # Configure scheduler
            if self.best_params["scheduler"] == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=10
                )
            elif self.best_params["scheduler"] == "plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5
                )
            elif self.best_params["scheduler"] == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.5
                )
            else:
                scheduler = None
            
            # Configure loss function
            criterion = nn.MSELoss()
            
            # Train for more epochs
            num_epochs = 50
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.best_model.to(device)
            
            for epoch in range(num_epochs):
                # Training
                self.best_model.train()
                train_loss = 0.0
                for batch_idx, (data, target) in enumerate(self.data_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = self.best_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(self.data_loader)
                
                # Validation
                self.best_model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in self.val_loader:
                        data, target = data.to(device), target.to(device)
                        output = self.best_model(data)
                        val_loss += criterion(output, target).item()
                        
                        # For directional accuracy
                        pred_direction = (output[:, 0] > 0).float()
                        true_direction = (target[:, 0] > 0).float()
                        correct += (pred_direction == true_direction).sum().item()
                        total += target.size(0)
                
                val_loss /= len(self.val_loader)
                val_accuracy = correct / total
                
                # Update scheduler if needed
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # Log metrics
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }, step=epoch)
            
            # Evaluate on test set if available
            if self.test_loader is not None:
                self.best_model.eval()
                test_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in self.test_loader:
                        data, target = data.to(device), target.to(device)
                        output = self.best_model(data)
                        test_loss += criterion(output, target).item()
                        
                        # For directional accuracy
                        pred_direction = (output[:, 0] > 0).float()
                        true_direction = (target[:, 0] > 0).float()
                        correct += (pred_direction == true_direction).sum().item()
                        total += target.size(0)
                
                test_loss /= len(self.test_loader)
                test_accuracy = correct / total
                
                # Log test metrics
                mlflow.log_metrics({
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy
                })
                
                logger.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
            
            # Save model
            mlflow.pytorch.log_model(self.best_model, "best_model")
            
            # Save as ONNX for deployment
            dummy_input = next(iter(self.data_loader))[0][0:1].to(device)
            mlflow.onnx.log_model(self.best_model, "onnx_model", input_example=dummy_input)
            
            logger.info("Best model training completed and saved")


def main():
    """Main function to run hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Cosmic Market Oracle")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--study-name", type=str, default="cosmic_market_oracle", help="Study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    args = parser.parse_args()
    
    # Load data
    # This is a placeholder - in a real implementation, you would load your data here
    # using the appropriate data loader for your project
    
    # Run hyperparameter tuning
    tuner = OptunaHyperparameterTuner(
        config_path=args.config,
        data_loader=None,  # Replace with actual data loader
        val_loader=None,   # Replace with actual validation loader
        test_loader=None,  # Replace with actual test loader
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        storage=args.storage
    )
    
    results = tuner.optimize()
    
    # Print results
    logger.info("Optimization completed!")
    logger.info(f"Best parameters: {results['best_params']}")
    logger.info(f"Best score: {results['best_score']}")


if __name__ == "__main__":
    main()
