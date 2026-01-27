"""
입자 군집 최적화 (Particle Swarm Optimization, PSO)

새나 물고기 떼의 군집 행동을 모방한 메타휴리스틱 최적화 알고리즘입니다.
"""

import numpy as np
from typing import Callable, Tuple, Optional


class ParticleSwarmOptimization:
    """
    입자 군집 최적화기
    
    각 입자는 자신의 최적 위치와 전역 최적 위치를 향해 이동합니다.
    
    Attributes
    ----------
    n_particles : int
        입자 수
    max_iter : int
        최대 반복 횟수
    w : float
        관성 가중치
    c1 : float
        인지 가중치 (개인 최적)
    c2 : float
        사회 가중치 (전역 최적)
    """
    
    def __init__(
        self,
        n_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        w_decay: float = 0.99
    ):
        """
        Parameters
        ----------
        n_particles : int
            입자 수 (기본값: 30)
        max_iter : int
            최대 반복 횟수 (기본값: 100)
        w : float
            관성 가중치 (기본값: 0.7)
        c1 : float
            인지 가중치 (기본값: 1.5)
        c2 : float
            사회 가중치 (기본값: 1.5)
        w_decay : float
            관성 가중치 감쇠율 (기본값: 0.99)
        """
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        
        self.history = {
            'best_f': [],
            'mean_f': [],
            'best_x': [],
            'iterations': 0
        }
    
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
            각 반복마다 호출할 콜백 함수
        
        Returns
        -------
        Tuple[np.ndarray, float]
            (최적 위치, 최적 함수값)
        """
        low, high = bounds
        
        # 초기화
        positions = np.random.uniform(low, high, (self.n_particles, dim))
        velocities = np.random.uniform(
            -(high - low) * 0.1,
            (high - low) * 0.1,
            (self.n_particles, dim)
        )
        
        # 개인 최적 (pbest)
        pbest_positions = positions.copy()
        pbest_scores = np.array([func(p) for p in positions])
        
        # 전역 최적 (gbest)
        gbest_idx = np.argmin(pbest_scores)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]
        
        self.history = {'best_f': [], 'mean_f': [], 'best_x': [], 'iterations': 0}
        
        w = self.w
        
        for i in range(self.max_iter):
            for j in range(self.n_particles):
                # 난수 생성
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)
                
                # 속도 업데이트
                cognitive = self.c1 * r1 * (pbest_positions[j] - positions[j])
                social = self.c2 * r2 * (gbest_position - positions[j])
                velocities[j] = w * velocities[j] + cognitive + social
                
                # 속도 제한
                v_max = (high - low) * 0.2
                velocities[j] = np.clip(velocities[j], -v_max, v_max)
                
                # 위치 업데이트
                positions[j] = positions[j] + velocities[j]
                positions[j] = np.clip(positions[j], low, high)
                
                # 적합도 평가
                score = func(positions[j])
                
                # 개인 최적 갱신
                if score < pbest_scores[j]:
                    pbest_scores[j] = score
                    pbest_positions[j] = positions[j].copy()
                    
                    # 전역 최적 갱신
                    if score < gbest_score:
                        gbest_score = score
                        gbest_position = positions[j].copy()
            
            # 관성 가중치 감쇠
            w *= self.w_decay
            
            # 이력 저장
            self.history['best_f'].append(gbest_score)
            self.history['mean_f'].append(np.mean(pbest_scores))
            self.history['best_x'].append(gbest_position.copy())
            
            if callback is not None:
                callback(i, gbest_position, gbest_score)
        
        self.history['iterations'] = self.max_iter
        
        return gbest_position, gbest_score
    
    def get_history(self) -> dict:
        """최적화 이력 반환"""
        return self.history
