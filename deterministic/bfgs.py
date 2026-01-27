"""
BFGS 준뉴턴 방법 (Broyden-Fletcher-Goldfarb-Shanno)

헤시안의 역행렬을 점진적으로 근사하여 뉴턴 방법의 계산 비용을 줄인 실용적인 알고리즘입니다.
"""

import numpy as np
from typing import Callable, Tuple, Optional


class BFGS:
    """
    BFGS 준뉴턴 최적화기
    
    헤시안의 역행렬을 랭크-2 업데이트로 근사합니다.
    실무에서 가장 널리 사용되는 준뉴턴 방법입니다.
    
    Attributes
    ----------
    max_iter : int
        최대 반복 횟수
    tol : float
        수렴 허용 오차
    c1 : float
        Armijo 조건 상수
    c2 : float
        Wolfe 조건 상수
    """
    
    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-6,
        c1: float = 1e-4,
        c2: float = 0.9
    ):
        """
        Parameters
        ----------
        max_iter : int
            최대 반복 횟수 (기본값: 500)
        tol : float
            수렴 허용 오차 (기본값: 1e-6)
        c1 : float
            Armijo 조건 상수 (기본값: 1e-4)
        c2 : float
            Wolfe 조건 상수 (기본값: 0.9)
        """
        self.max_iter = max_iter
        self.tol = tol
        self.c1 = c1
        self.c2 = c2
        
        self.history = {
            'x': [],
            'f': [],
            'grad_norm': [],
            'iterations': 0
        }
    
    def _line_search_wolfe(
        self,
        func: Callable,
        grad_func: Callable,
        x: np.ndarray,
        direction: np.ndarray,
        max_iter: int = 50
    ) -> float:
        """
        Wolfe 조건을 만족하는 스텝 크기 탐색
        
        Parameters
        ----------
        func : Callable
            목적 함수
        grad_func : Callable
            그래디언트 함수
        x : np.ndarray
            현재 위치
        direction : np.ndarray
            탐색 방향
        max_iter : int
            최대 반복 횟수
        
        Returns
        -------
        float
            스텝 크기
        """
        alpha = 1.0
        f_x = func(x)
        grad_x = grad_func(x)
        slope = np.dot(grad_x, direction)
        
        for _ in range(max_iter):
            x_new = x + alpha * direction
            f_new = func(x_new)
            
            # Armijo 조건 확인
            if f_new > f_x + self.c1 * alpha * slope:
                alpha *= 0.5
                continue
            
            # Curvature 조건 확인
            grad_new = grad_func(x_new)
            if np.dot(grad_new, direction) < self.c2 * slope:
                alpha *= 1.5
                continue
            
            return alpha
        
        return alpha
    
    def optimize(
        self,
        func: Callable,
        grad_func: Callable,
        x0: np.ndarray,
        callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, float]:
        """
        최적화 수행
        
        Parameters
        ----------
        func : Callable
            목적 함수 f(x)
        grad_func : Callable
            그래디언트 함수 ∇f(x)
        x0 : np.ndarray
            초기 위치
        callback : Callable, optional
            각 반복마다 호출할 콜백 함수
        
        Returns
        -------
        Tuple[np.ndarray, float]
            (최적 위치, 최적 함수값)
        """
        x = np.asarray(x0, dtype=float).copy()
        n = len(x)
        
        # 초기 헤시안 역행렬 근사 (단위 행렬)
        H = np.eye(n)
        
        self.history = {'x': [], 'f': [], 'grad_norm': [], 'iterations': 0}
        
        grad = grad_func(x)
        
        for i in range(self.max_iter):
            grad_norm = np.linalg.norm(grad)
            f_val = func(x)
            
            self.history['x'].append(x.copy())
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            
            # 수렴 확인
            if grad_norm < self.tol:
                break
            
            # 탐색 방향 계산: p = -H * ∇f
            direction = -H @ grad
            
            # 선탐색으로 스텝 크기 결정
            alpha = self._line_search_wolfe(func, grad_func, x, direction)
            
            # 스텝 계산
            s = alpha * direction
            x_new = x + s
            
            # 그래디언트 차이 계산
            grad_new = grad_func(x_new)
            y = grad_new - grad
            
            # BFGS 업데이트
            sy = np.dot(s, y)
            
            if sy > 1e-10:  # 양의 정부호 확인
                rho = 1.0 / sy
                I = np.eye(n)
                
                # H_{k+1} = (I - ρsy^T) H_k (I - ρys^T) + ρss^T
                s = s.reshape(-1, 1)
                y = y.reshape(-1, 1)
                
                V = I - rho * (s @ y.T)
                H = V @ H @ V.T + rho * (s @ s.T)
            
            # 업데이트
            x = x_new
            grad = grad_new
            
            if callback is not None:
                callback(i, x, f_val, grad_norm)
        
        self.history['iterations'] = i + 1
        
        return x, func(x)
    
    def get_history(self) -> dict:
        """최적화 이력 반환"""
        return self.history
