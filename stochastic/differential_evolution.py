"""
차분 진화 (Differential Evolution)

벡터의 차분을 활용한 진화 알고리즘으로, 연속 최적화에 강력합니다.
"""

import numpy as np
from typing import Callable, Tuple, Optional


class DifferentialEvolution:
    """
    차분 진화 최적화기
    
    DE/rand/1/bin 전략을 기본으로 사용합니다.
    
    Attributes
    ----------
    pop_size : int
        집단 크기
    max_generations : int
        최대 세대 수
    F : float
        스케일링 인자 (돌연변이 강도)
    CR : float
        교차 확률
    """
    
    def __init__(
        self,
        pop_size: int = 50,
        max_generations: int = 100,
        F: float = 0.8,
        CR: float = 0.9,
        strategy: str = 'rand1bin'
    ):
        """
        Parameters
        ----------
        pop_size : int
            집단 크기 (기본값: 50)
        max_generations : int
            최대 세대 수 (기본값: 100)
        F : float
            스케일링 인자 (기본값: 0.8)
        CR : float
            교차 확률 (기본값: 0.9)
        strategy : str
            진화 전략 (기본값: 'rand1bin')
        """
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR
        self.strategy = strategy
        
        self.history = {
            'best_f': [],
            'mean_f': [],
            'best_x': [],
            'generations': 0
        }
    
    def _mutation_rand1(
        self,
        population: np.ndarray,
        idx: int
    ) -> np.ndarray:
        """
        DE/rand/1 돌연변이
        
        v = x_r1 + F * (x_r2 - x_r3)
        """
        idxs = list(range(self.pop_size))
        idxs.remove(idx)
        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
        
        return population[r1] + self.F * (population[r2] - population[r3])
    
    def _mutation_best1(
        self,
        population: np.ndarray,
        best_idx: int,
        idx: int
    ) -> np.ndarray:
        """
        DE/best/1 돌연변이
        
        v = x_best + F * (x_r1 - x_r2)
        """
        idxs = list(range(self.pop_size))
        idxs.remove(idx)
        if best_idx in idxs:
            idxs.remove(best_idx)
        r1, r2 = np.random.choice(idxs, 2, replace=False)
        
        return population[best_idx] + self.F * (population[r1] - population[r2])
    
    def _crossover_bin(
        self,
        target: np.ndarray,
        mutant: np.ndarray
    ) -> np.ndarray:
        """
        이항 교차 (Binomial Crossover)
        """
        dim = len(target)
        trial = target.copy()
        
        # 최소 하나는 mutant에서 가져옴
        j_rand = np.random.randint(dim)
        
        for j in range(dim):
            if np.random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        
        return trial
    
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
        low, high = bounds
        
        # 초기화
        population = np.random.uniform(low, high, (self.pop_size, dim))
        fitness = np.array([func(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]
        
        self.history = {'best_f': [], 'mean_f': [], 'best_x': [], 'generations': 0}
        
        for gen in range(self.max_generations):
            for i in range(self.pop_size):
                # 돌연변이
                if self.strategy == 'best1bin':
                    mutant = self._mutation_best1(population, best_idx, i)
                else:  # 'rand1bin'
                    mutant = self._mutation_rand1(population, i)
                
                # 경계 처리
                mutant = np.clip(mutant, low, high)
                
                # 교차
                trial = self._crossover_bin(population[i], mutant)
                
                # 선택
                trial_fitness = func(trial)
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_f:
                        best_x = trial.copy()
                        best_f = trial_fitness
                        best_idx = i
            
            # 이력 저장
            self.history['best_f'].append(best_f)
            self.history['mean_f'].append(np.mean(fitness))
            self.history['best_x'].append(best_x.copy())
            
            if callback is not None:
                callback(gen, best_x, best_f)
        
        self.history['generations'] = self.max_generations
        
        return best_x, best_f
    
    def get_history(self) -> dict:
        """최적화 이력 반환"""
        return self.history
