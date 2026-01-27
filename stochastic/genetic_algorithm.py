"""
유전 알고리즘 (Genetic Algorithm)

자연선택과 유전학의 원리를 모방한 메타휴리스틱 최적화 알고리즘입니다.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List


class GeneticAlgorithm:
    """
    유전 알고리즘 최적화기
    
    선택, 교차, 돌연변이 연산자를 통해 해 집단을 진화시킵니다.
    
    Attributes
    ----------
    pop_size : int
        집단 크기
    max_generations : int
        최대 세대 수
    crossover_rate : float
        교차 확률
    mutation_rate : float
        돌연변이 확률
    mutation_scale : float
        돌연변이 크기
    """
    
    def __init__(
        self,
        pop_size: int = 50,
        max_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.1,
        elitism: int = 2
    ):
        """
        Parameters
        ----------
        pop_size : int
            집단 크기 (기본값: 50)
        max_generations : int
            최대 세대 수 (기본값: 100)
        crossover_rate : float
            교차 확률 (기본값: 0.8)
        mutation_rate : float
            돌연변이 확률 (기본값: 0.1)
        mutation_scale : float
            돌연변이 크기 (기본값: 0.1)
        elitism : int
            엘리트 보존 개체 수 (기본값: 2)
        """
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elitism = elitism
        
        self.history = {
            'best_f': [],
            'mean_f': [],
            'best_x': [],
            'generations': 0
        }
    
    def _initialize_population(
        self,
        dim: int,
        bounds: Tuple[float, float]
    ) -> np.ndarray:
        """
        초기 집단 생성
        
        Parameters
        ----------
        dim : int
            차원
        bounds : Tuple[float, float]
            탐색 범위 (최소, 최대)
        
        Returns
        -------
        np.ndarray
            초기 집단 (pop_size x dim)
        """
        low, high = bounds
        return np.random.uniform(low, high, (self.pop_size, dim))
    
    def _evaluate_fitness(
        self,
        population: np.ndarray,
        func: Callable
    ) -> np.ndarray:
        """
        적합도 평가 (최솟값 문제를 위해 음수화)
        """
        fitness = np.array([func(ind) for ind in population])
        # 최솟값 문제이므로 음수로 변환 후 최댓값 선택
        return -fitness
    
    def _selection_tournament(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        tournament_size: int = 3
    ) -> np.ndarray:
        """
        토너먼트 선택
        """
        selected = []
        for _ in range(self.pop_size):
            candidates = np.random.choice(self.pop_size, tournament_size, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            selected.append(population[winner].copy())
        return np.array(selected)
    
    def _crossover_blend(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        블렌드 교차 (BLX-α)
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        diff = np.abs(parent1 - parent2)
        low = np.minimum(parent1, parent2) - alpha * diff
        high = np.maximum(parent1, parent2) + alpha * diff
        
        child1 = np.random.uniform(low, high)
        child2 = np.random.uniform(low, high)
        
        return child1, child2
    
    def _mutation_gaussian(
        self,
        individual: np.ndarray,
        bounds: Tuple[float, float]
    ) -> np.ndarray:
        """
        가우시안 돌연변이
        """
        low, high = bounds
        range_size = high - low
        
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                individual[i] += np.random.normal(0, self.mutation_scale * range_size)
                individual[i] = np.clip(individual[i], low, high)
        
        return individual
    
    def optimize(
        self,
        func: Callable,
        dim: int,
        bounds: Tuple[float, float],
        callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, float]:
        """
        최적화 수행
        
        Parameters
        ----------
        func : Callable
            목적 함수 f(x)
        dim : int
            차원
        bounds : Tuple[float, float]
            탐색 범위
        callback : Callable, optional
            각 세대마다 호출할 콜백 함수
        
        Returns
        -------
        Tuple[np.ndarray, float]
            (최적 위치, 최적 함수값)
        """
        # 초기화
        population = self._initialize_population(dim, bounds)
        
        self.history = {'best_f': [], 'mean_f': [], 'best_x': [], 'generations': 0}
        
        best_individual = None
        best_fitness = float('-inf')
        
        for gen in range(self.max_generations):
            # 적합도 평가
            fitness = self._evaluate_fitness(population, func)
            
            # 최고 개체 갱신
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            # 이력 저장
            self.history['best_f'].append(-best_fitness)
            self.history['mean_f'].append(-np.mean(fitness))
            self.history['best_x'].append(best_individual.copy())
            
            # 엘리트 보존
            elite_indices = np.argsort(fitness)[-self.elitism:]
            elites = population[elite_indices].copy()
            
            # 선택
            selected = self._selection_tournament(population, fitness)
            
            # 교차 및 돌연변이
            new_population = []
            for i in range(0, self.pop_size - self.elitism, 2):
                if i + 1 < self.pop_size - self.elitism:
                    child1, child2 = self._crossover_blend(selected[i], selected[i + 1])
                    child1 = self._mutation_gaussian(child1, bounds)
                    child2 = self._mutation_gaussian(child2, bounds)
                    new_population.extend([child1, child2])
                else:
                    child = self._mutation_gaussian(selected[i].copy(), bounds)
                    new_population.append(child)
            
            # 엘리트 추가
            new_population = np.array(new_population[:self.pop_size - self.elitism])
            population = np.vstack([new_population, elites])
            
            if callback is not None:
                callback(gen, best_individual, -best_fitness)
        
        self.history['generations'] = self.max_generations
        
        return best_individual, -best_fitness
    
    def get_history(self) -> dict:
        """최적화 이력 반환"""
        return self.history
