"""
Genetic Feature Discovery Module for the Cosmic Market Oracle.

This module implements genetic programming techniques for discovering
optimal combinations of astrological factors for financial market prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, Union
from datetime import datetime, timedelta
import random
from enum import Enum
import copy
import logging
from dataclasses import dataclass, field

# Import from the centralized feature_definitions module
from .feature_definitions import FeatureDefinition
from .feature_generator import FeatureGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GeneticIndividual:
    """Individual in the genetic algorithm population."""
    feature_def: FeatureDefinition
    fitness: float = 0.0
    market_correlation: float = 0.0
    complexity: int = 0
    generation: int = 0


class GeneticFeatureDiscovery:
    """Genetic programming for astrological feature discovery."""
    
    def __init__(self, feature_generator: FeatureGenerator, financial_data_provider=None,
                population_size: int = 50, generations: int = 20,
                mutation_rate: float = 0.3, crossover_rate: float = 0.7):
        """
        Initialize the genetic feature discovery.
        
        Args:
            feature_generator: Feature generator instance
            financial_data_provider: Provider of financial data (optional)
            population_size: Size of the genetic population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.feature_generator = feature_generator
        self.financial_data_provider = financial_data_provider
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initialize population
        self.population = []
        
        # Initialize best individuals
        self.best_individuals = []
        
        # Initialize feature history
        self.feature_history = {}
    
    def initialize_population(self):
        """Initialize the genetic population with random individuals."""
        self.population = []
        
        for _ in range(self.population_size):
            # Generate random feature definition
            feature_def = self.feature_generator.generate_feature_definition(
                transformation_type=random.choice([None] + self.feature_generator.transformation_types),
                combination_type=random.choice([None] + self.feature_generator.combination_types)
            )
            
            # Create individual
            individual = GeneticIndividual(
                feature_def=feature_def,
                complexity=self._calculate_complexity(feature_def),
                generation=0
            )
            
            self.population.append(individual)
            
            # Add to feature history
            self.feature_history[feature_def.name] = {
                "definition": feature_def,
                "generation": 0,
                "fitness": 0.0,
                "market_correlation": 0.0,
                "complexity": individual.complexity
            }
        
        logger.info(f"Initialized population with {len(self.population)} individuals")
    
    def evolve(self, start_date: datetime, end_date: datetime, target_symbol: str = "SPY"):
        """
        Evolve the population to discover optimal features.
        
        Args:
            start_date: Start date for evaluation
            end_date: End date for evaluation
            target_symbol: Financial symbol to target
        """
        # Initialize population if empty
        if not self.population:
            self.initialize_population()
        
        # Get financial data
        financial_data = None
        if self.financial_data_provider:
            financial_data = self.financial_data_provider.get_historical_data(
                target_symbol, start_date, end_date
            )
        
        # Evolve for specified number of generations
        for generation in range(self.generations):
            logger.info(f"Starting generation {generation+1}/{self.generations}")
            
            # Evaluate fitness
            self._evaluate_fitness(start_date, end_date, financial_data, target_symbol)
            
            # Select best individuals
            self._update_best_individuals()
            
            # Create next generation
            next_generation = self._create_next_generation()
            
            # Update population
            self.population = next_generation
            
            # Log progress
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            logger.info(f"Generation {generation+1} best fitness: {best_individual.fitness:.4f}")
            logger.info(f"Best feature: {best_individual.feature_def.name}")
        
        # Final evaluation
        self._evaluate_fitness(start_date, end_date, financial_data, target_symbol)
        self._update_best_individuals()
        
        logger.info("Evolution complete")
        logger.info(f"Discovered {len(self.best_individuals)} high-quality features")
    
    def _evaluate_fitness(self, start_date: datetime, end_date: datetime, 
                         financial_data: pd.DataFrame, target_symbol: str):
        """
        Evaluate the fitness of all individuals in the population.
        
        Args:
            start_date: Start date for evaluation
            end_date: End date for evaluation
            financial_data: Financial data for evaluation
            target_symbol: Financial symbol to target
        """
        # Calculate feature values for each date
        feature_values = {}
        
        # Generate dates
        current_date = start_date
        dates = []
        
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Get base features for all dates
        base_features = {}
        for date in dates:
            base_features[date] = self.feature_generator.generate_base_features(date)
        
        # Calculate feature values for each individual
        for individual in self.population:
            feature_def = individual.feature_def
            feature_name = feature_def.name
            
            feature_values[feature_name] = []
            
            for date in dates:
                value = self.feature_generator.calculate_feature_value(
                    feature_def, date, base_features[date]
                )
                feature_values[feature_name].append(value)
        
        # Calculate fitness based on market correlation if financial data is available
        if financial_data is not None and len(financial_data) > 0:
            # Align dates with financial data
            aligned_dates = []
            for date in dates:
                date_str = date.strftime("%Y-%m-%d")
                if date_str in financial_data.index:
                    aligned_dates.append(date)
            
            # Calculate returns
            returns = financial_data["close"].pct_change().dropna()
            
            # Calculate correlation with market returns
            for individual in self.population:
                feature_name = individual.feature_def.name
                
                # Get feature values for aligned dates
                aligned_values = []
                for date in aligned_dates:
                    date_idx = dates.index(date)
                    if date_idx < len(feature_values[feature_name]):
                        aligned_values.append(feature_values[feature_name][date_idx])
                
                if len(aligned_values) > 0 and len(returns) > 0:
                    # Calculate correlation
                    correlation = np.corrcoef(aligned_values[:len(returns)], returns.values[:len(aligned_values)])[0, 1]
                    
                    # Handle NaN correlation
                    if np.isnan(correlation):
                        correlation = 0.0
                    
                    # Update individual
                    individual.market_correlation = abs(correlation)
                    
                    # Calculate fitness
                    complexity_penalty = individual.complexity / 10.0
                    individual.fitness = abs(correlation) - complexity_penalty
                    
                    # Update feature history
                    if feature_name in self.feature_history:
                        self.feature_history[feature_name]["fitness"] = individual.fitness
                        self.feature_history[feature_name]["market_correlation"] = individual.market_correlation
        else:
            # Without financial data, use complexity as a proxy for fitness
            for individual in self.population:
                complexity_score = 1.0 / (1.0 + individual.complexity)
                individual.fitness = complexity_score
    
    def _update_best_individuals(self, max_best: int = 10):
        """
        Update the list of best individuals.
        
        Args:
            max_best: Maximum number of best individuals to keep
        """
        # Sort population by fitness
        sorted_population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        
        # Update best individuals
        for individual in sorted_population[:max_best]:
            # Check if already in best individuals
            if not any(best.feature_def.name == individual.feature_def.name for best in self.best_individuals):
                self.best_individuals.append(copy.deepcopy(individual))
        
        # Sort and limit best individuals
        self.best_individuals = sorted(self.best_individuals, key=lambda ind: ind.fitness, reverse=True)[:max_best]
    
    def _create_next_generation(self):
        """
        Create the next generation through selection, crossover, and mutation.
        
        Returns:
            List of individuals for the next generation
        """
        next_generation = []
        
        # Elitism: Keep best individuals
        elites = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)[:max(1, int(self.population_size * 0.1))]
        for elite in elites:
            next_generation.append(copy.deepcopy(elite))
        
        # Fill the rest through selection, crossover, and mutation
        while len(next_generation) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            
            if random.random() < self.crossover_rate and len(self.population) > 1:
                # Crossover
                parent2 = self._tournament_selection()
                while parent2.feature_def.name == parent1.feature_def.name:
                    parent2 = self._tournament_selection()
                
                child = self._crossover(parent1, parent2)
            else:
                # Cloning
                child = copy.deepcopy(parent1)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            # Update generation
            child.generation = max(parent1.generation, 0) + 1
            child.feature_def.generation = child.generation
            
            # Add to next generation
            next_generation.append(child)
            
            # Add to feature history
            self.feature_history[child.feature_def.name] = {
                "definition": child.feature_def,
                "generation": child.generation,
                "fitness": child.fitness,
                "market_correlation": child.market_correlation,
                "complexity": child.complexity
            }
        
        return next_generation
    
    def _tournament_selection(self, tournament_size: int = 3):
        """
        Select an individual using tournament selection.
        
        Args:
            tournament_size: Number of individuals in the tournament
            
        Returns:
            Selected individual
        """
        # Select random individuals for the tournament
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        
        # Return the best individual from the tournament
        return max(tournament, key=lambda ind: ind.fitness)
    
    def _crossover(self, parent1: GeneticIndividual, parent2: GeneticIndividual) -> GeneticIndividual:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child individual
        """
        # Create a new feature definition
        feature_type = random.choice([parent1.feature_def.feature_type, parent2.feature_def.feature_type])
        
        # Combine parameters
        parameters = {}
        for key in set(parent1.feature_def.parameters.keys()) | set(parent2.feature_def.parameters.keys()):
            if key in parent1.feature_def.parameters and key in parent2.feature_def.parameters:
                # Choose randomly between parents
                parameters[key] = random.choice([parent1.feature_def.parameters[key], parent2.feature_def.parameters[key]])
            elif key in parent1.feature_def.parameters:
                parameters[key] = parent1.feature_def.parameters[key]
            else:
                parameters[key] = parent2.feature_def.parameters[key]
        
        # Create name and description
        name = f"crossover_{parent1.feature_def.name[:10]}_{parent2.feature_def.name[:10]}_{random.randint(1000, 9999)}"
        description = f"Crossover of {parent1.feature_def.name} and {parent2.feature_def.name}"
        
        # Create feature definition
        feature_def = FeatureDefinition(
            name=name,
            description=description,
            feature_type=feature_type,
            parameters=parameters,
            parent_features=[parent1.feature_def.name, parent2.feature_def.name]
        )
        
        # Create child individual
        child = GeneticIndividual(
            feature_def=feature_def,
            complexity=self._calculate_complexity(feature_def)
        )
        
        return child
    
    def _mutate(self, individual: GeneticIndividual) -> GeneticIndividual:
        """
        Perform mutation on an individual.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        # Clone the individual
        mutated = copy.deepcopy(individual)
        
        # Choose mutation type
        mutation_type = random.choice([
            "add_transformation",
            "change_transformation",
            "add_combination",
            "change_combination",
            "modify_parameters"
        ])
        
        if mutation_type == "add_transformation" and "transformation" not in mutated.feature_def.parameters:
            # Add a transformation
            transformation_type = random.choice(self.feature_generator.transformation_types)
            
            # Generate transformation parameters
            transform_parameters = {}
            
            if transformation_type == "sine_transform" or transformation_type == "cosine_transform":
                transform_parameters["scale"] = random.uniform(0.5, 2.0)
            
            elif transformation_type == "threshold":
                transform_parameters["threshold"] = random.uniform(0.2, 0.8)
            
            elif transformation_type == "binary_encoding":
                transform_parameters["thresholds"] = [random.uniform(0.2, 0.4), random.uniform(0.6, 0.8)]
            
            # Update parameters and name
            mutated.feature_def.parameters["transformation"] = {
                "type": transformation_type,
                "parameters": transform_parameters
            }
            
            mutated.feature_def.name = f"{mutated.feature_def.name}_{transformation_type}"
            mutated.feature_def.description = f"{mutated.feature_def.description} with {transformation_type} transformation"
        
        elif mutation_type == "change_transformation" and "transformation" in mutated.feature_def.parameters:
            # Change transformation type
            current_type = mutated.feature_def.parameters["transformation"]["type"]
            new_types = [t for t in self.feature_generator.transformation_types if t != current_type]
            
            if new_types:
                new_type = random.choice(new_types)
                
                # Generate new parameters
                transform_parameters = {}
                
                if new_type == "sine_transform" or new_type == "cosine_transform":
                    transform_parameters["scale"] = random.uniform(0.5, 2.0)
                
                elif new_type == "threshold":
                    transform_parameters["threshold"] = random.uniform(0.2, 0.8)
                
                elif new_type == "binary_encoding":
                    transform_parameters["thresholds"] = [random.uniform(0.2, 0.4), random.uniform(0.6, 0.8)]
                
                # Update parameters and name
                mutated.feature_def.parameters["transformation"]["type"] = new_type
                mutated.feature_def.parameters["transformation"]["parameters"] = transform_parameters
                
                mutated.feature_def.name = mutated.feature_def.name.replace(current_type, new_type)
                mutated.feature_def.description = mutated.feature_def.description.replace(current_type, new_type)
        
        elif mutation_type == "add_combination" and "combination" not in mutated.feature_def.parameters:
            # Add a combination
            combination_type = random.choice(self.feature_generator.combination_types)
            
            # Generate combination parameters
            combine_parameters = {}
            
            if combination_type == "weighted_sum":
                combine_parameters["weights"] = [random.uniform(0.1, 1.0) for _ in range(3)]
            
            elif combination_type == "conditional":
                combine_parameters["threshold"] = random.uniform(0.2, 0.8)
            
            # Update parameters and name
            mutated.feature_def.parameters["combination"] = {
                "type": combination_type,
                "parameters": combine_parameters
            }
            
            mutated.feature_def.name = f"{mutated.feature_def.name}_{combination_type}"
            mutated.feature_def.description = f"{mutated.feature_def.description} with {combination_type} combination"
            
            # Add parent features
            if not mutated.feature_def.parent_features:
                # Select random features from catalog
                catalog_features = list(self.feature_generator.feature_catalog.keys())
                if catalog_features:
                    parent_features = random.sample(catalog_features, min(2, len(catalog_features)))
                    mutated.feature_def.parent_features = parent_features
        
        elif mutation_type == "change_combination" and "combination" in mutated.feature_def.parameters:
            # Change combination type
            current_type = mutated.feature_def.parameters["combination"]["type"]
            new_types = [t for t in self.feature_generator.combination_types if t != current_type]
            
            if new_types:
                new_type = random.choice(new_types)
                
                # Generate new parameters
                combine_parameters = {}
                
                if new_type == "weighted_sum":
                    combine_parameters["weights"] = [random.uniform(0.1, 1.0) for _ in range(3)]
                
                elif new_type == "conditional":
                    combine_parameters["threshold"] = random.uniform(0.2, 0.8)
                
                # Update parameters and name
                mutated.feature_def.parameters["combination"]["type"] = new_type
                mutated.feature_def.parameters["combination"]["parameters"] = combine_parameters
                
                mutated.feature_def.name = mutated.feature_def.name.replace(current_type, new_type)
                mutated.feature_def.description = mutated.feature_def.description.replace(current_type, new_type)
        
        elif mutation_type == "modify_parameters":
            # Modify existing parameters
            if "transformation" in mutated.feature_def.parameters:
                transform_type = mutated.feature_def.parameters["transformation"]["type"]
                
                if transform_type == "sine_transform" or transform_type == "cosine_transform":
                    mutated.feature_def.parameters["transformation"]["parameters"]["scale"] = random.uniform(0.5, 2.0)
                
                elif transform_type == "threshold":
                    mutated.feature_def.parameters["transformation"]["parameters"]["threshold"] = random.uniform(0.2, 0.8)
                
                elif transform_type == "binary_encoding":
                    mutated.feature_def.parameters["transformation"]["parameters"]["thresholds"] = [
                        random.uniform(0.2, 0.4), random.uniform(0.6, 0.8)
                    ]
            
            if "combination" in mutated.feature_def.parameters:
                combine_type = mutated.feature_def.parameters["combination"]["type"]
                
                if combine_type == "weighted_sum":
                    mutated.feature_def.parameters["combination"]["parameters"]["weights"] = [
                        random.uniform(0.1, 1.0) for _ in range(3)
                    ]
                
                elif combine_type == "conditional":
                    mutated.feature_def.parameters["combination"]["parameters"]["threshold"] = random.uniform(0.2, 0.8)
        
        # Update name to indicate mutation
        mutated.feature_def.name = f"mutated_{mutated.feature_def.name}"
        
        # Update complexity
        mutated.complexity = self._calculate_complexity(mutated.feature_def)
        
        return mutated
    
    def _calculate_complexity(self, feature_def: FeatureDefinition) -> int:
        """
        Calculate the complexity of a feature definition.
        
        Args:
            feature_def: Feature definition
            
        Returns:
            Complexity score
        """
        complexity = 1  # Base complexity
        
        # Add complexity for transformations
        if "transformation" in feature_def.parameters:
            complexity += 1
        
        # Add complexity for combinations
        if "combination" in feature_def.parameters:
            complexity += 1
            
            # Add complexity for parent features
            complexity += len(feature_def.parent_features)
        
        return complexity
    
    def get_best_features(self, top_n: int = 10) -> List[FeatureDefinition]:
        """
        Get the best features discovered.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of best feature definitions
        """
        # Sort best individuals by fitness
        sorted_individuals = sorted(self.best_individuals, key=lambda ind: ind.fitness, reverse=True)
        
        # Return feature definitions
        return [ind.feature_def for ind in sorted_individuals[:top_n]]
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance metrics.
        
        Args:
            top_n: Number of top features to include
            
        Returns:
            DataFrame with feature importance metrics
        """
        # Create feature importance data
        importance_data = []
        
        for feature_name, feature_info in self.feature_history.items():
            importance_data.append({
                "feature_name": feature_name,
                "fitness": feature_info["fitness"],
                "market_correlation": feature_info["market_correlation"],
                "complexity": feature_info["complexity"],
                "generation": feature_info["generation"]
            })
        
        # Convert to DataFrame
        importance_df = pd.DataFrame(importance_data)
        
        # Sort by fitness
        importance_df = importance_df.sort_values("fitness", ascending=False)
        
        # Return top N
        return importance_df.head(top_n)


# Example usage
if __name__ == "__main__":
    from src.astro_engine.astronomical_calculator import AstronomicalCalculator
    from src.data_acquisition.financial_data import FinancialDataProvider
    
    # Initialize calculator
    calculator = AstronomicalCalculator()
    
    # Initialize feature generator
    feature_generator = FeatureGenerator(calculator)
    
    # Initialize financial data provider
    financial_provider = FinancialDataProvider()
    
    # Initialize genetic feature discovery
    genetic_discovery = GeneticFeatureDiscovery(
        feature_generator,
        financial_provider,
        population_size=30,
        generations=10
    )
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Evolve to discover features
    genetic_discovery.evolve(start_date, end_date, "SPY")
    
    # Get best features
    best_features = genetic_discovery.get_best_features(5)
    
    print("Top 5 discovered features:")
    for i, feature in enumerate(best_features):
        print(f"{i+1}. {feature.name}")
        print(f"   Description: {feature.description}")
        print(f"   Generation: {feature.generation}")
        print(f"   Parameters: {feature.parameters}")
        print()
    
    # Get feature importance
    importance_df = genetic_discovery.get_feature_importance(10)
    print("Feature importance:")
    print(importance_df)
