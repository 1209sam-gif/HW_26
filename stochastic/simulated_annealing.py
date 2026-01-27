"""
담금질 기법 (Simulated Annealing)

금속의 열처리 과정을 모방한 메타휴리스틱 최적화 알고리즘입니다.
"""

import numpy as np
from typing import Callable, Tuple, Optional


class SimulatedAnnealing:
    """
    담금질 기법 최적화기
    
    높은 온도에서 시작하여 점진적으로 냉각하면서 최적해를 탐색합니다.
    지역 최솟값에서 빠져나올 확률이 온도에 비례합니다.
    
    Attributes
    ----------
    initial_temp : float
        초기 온도
    final_temp : float
        최종 온도
    cooling_rate : float
        냉각률
    max_iter : int
        각 온도에서의 최대 반복 횟수
    """
    
    def __init__(
        self,
        initial_temp: float = 100.0,
        final_temp: float = 0.001,
        cooling_rate: float = 0.95,
        max_iter_per_temp: int = 50
    ):
        """
        Parameters
        ----------
        initial_temp : float
            초기 온도 (기본값: 100.0)
        final_temp : float
            최종 온도 (기본값: 0.001)
        cooling_rate : float
            냉각률 (기본값: 0.95)
        max_iter_per_temp : int
            각 온도에서의 최대 반복 횟수 (기본값: 50)
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iter_per_temp = max_iter_per_temp
        
        self.history = {
            'f': [],
            'best_f': [],
            'x': [],
            'temperature': [],
            'iterations': 0
        }
    
    def _neighbor(
        self,
        x: np.ndarray,
        bounds: Tuple[float, float],
        temp: float,
        initial_temp: float
    ) -> np.ndarray:
        """
        이웃 해 생성 (적응형 스텝 크기)
        """
        low, high = bounds
        range_size = high - low
        
        # 온도에 비례한 스텝 크기
        scale = (temp / initial_temp) * range_size * 0.1
        
        neighbor = x + np.random.normal(0, scale, len(x))
        neighbor = np.clip(neighbor, low, high)
        
        return neighbor
    
    def _acceptance_probability(
        self,
        current_cost: float,
        new_cost: float,
        temperature: float
    ) -> float:
        """
        수락 확률 계산 (Metropolis 기준)
        """
        if new_cost < current_cost:
            return 1.0
        
        return np.exp(-(new_cost - current_cost) / temperature)
    
    def optimize(
        self,
        func: Callable,
        dim: int,
        bounds: Tuple[float, float],
        x0: Optional[np.ndarray] = None,
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
        x0 : np.ndarray, optional
            초기 위치 (없으면 무작위)
        callback : Callable, optional
            각 반복마다 호출할 콜백 함수
        
        Returns
        -------
        Tuple[np.ndarray, float]
            (최적 위치, 최적 함수값)
        """
        low, high = bounds
        
        # 초기화
        if x0 is None:
            current_x = np.random.uniform(low, high, dim)
        else:
            current_x = np.asarray(x0).copy()
        
        current_cost = func(current_x)
        
        best_x = current_x.copy()
        best_cost = current_cost
        
        self.history = {
            'f': [],
            'best_f': [],
            'x': [],
            'temperature': [],
            'iterations': 0
        }
        
        temperature = self.initial_temp
        total_iterations = 0
        
        while temperature > self.final_temp:
            for _ in range(self.max_iter_per_temp):
                # 이웃 해 생성
                new_x = self._neighbor(
                    current_x, bounds, temperature, self.initial_temp
                )
                new_cost = func(new_x)
                
                # 수락 여부 결정
                if np.random.random() < self._acceptance_probability(
                    current_cost, new_cost, temperature
                ):
                    current_x = new_x
                    current_cost = new_cost
                    
                    # 최적해 갱신
                    if current_cost < best_cost:
                        best_x = current_x.copy()
                        best_cost = current_cost
                
                # 이력 저장
                self.history['f'].append(current_cost)
                self.history['best_f'].append(best_cost)
                self.history['x'].append(current_x.copy())
                self.history['temperature'].append(temperature)
                
                total_iterations += 1
            
            # 냉각
            temperature *= self.cooling_rate
            
            if callback is not None:
                callback(total_iterations, best_x, best_cost, temperature)
        
        self.history['iterations'] = total_iterations
        
        return best_x, best_cost
    
    def get_history(self) -> dict:
        """최적화 이력 반환"""
        return self.history
