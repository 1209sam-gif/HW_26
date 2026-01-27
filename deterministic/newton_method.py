"""
뉴턴 방법 (Newton's Method)

2차 도함수 정보(헤시안)를 활용하여 빠른 수렴을 달성하는 최적화 알고리즘입니다.
"""

import numpy as np
from typing import Callable, Tuple, Optional


class NewtonMethod:
    """
    뉴턴 방법 최적화기
    
    헤시안 행렬을 사용하여 2차 근사를 수행합니다.
    헤시안이 제공되지 않으면 유한 차분으로 근사합니다.
    
    Attributes
    ----------
    max_iter : int
        최대 반복 횟수
    tol : float
        수렴 허용 오차
    damping : float
        감쇠 계수 (특이 헤시안 방지)
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        damping: float = 1e-6
    ):
        """
        Parameters
        ----------
        max_iter : int
            최대 반복 횟수 (기본값: 100)
        tol : float
            수렴 허용 오차 (기본값: 1e-6)
        damping : float
            헤시안 정규화용 감쇠 계수 (기본값: 1e-6)
        """
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        
        self.history = {
            'x': [],
            'f': [],
            'grad_norm': [],
            'iterations': 0
        }
    
    def _numerical_hessian(
        self,
        grad_func: Callable,
        x: np.ndarray,
        epsilon: float = 1e-5
    ) -> np.ndarray:
        """
        유한 차분을 이용한 헤시안 근사
        
        Parameters
        ----------
        grad_func : Callable
            그래디언트 함수
        x : np.ndarray
            현재 위치
        epsilon : float
            유한 차분 스텝 크기
        
        Returns
        -------
        np.ndarray
            헤시안 행렬 근사
        """
        n = len(x)
        hessian = np.zeros((n, n))
        
        grad_x = grad_func(x)
        
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += epsilon
            grad_plus = grad_func(x_plus)
            
            hessian[:, i] = (grad_plus - grad_x) / epsilon
        
        # 대칭화
        hessian = (hessian + hessian.T) / 2
        
        return hessian
    
    def optimize(
        self,
        func: Callable,
        grad_func: Callable,
        x0: np.ndarray,
        hess_func: Optional[Callable] = None,
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
        hess_func : Callable, optional
            헤시안 함수 (없으면 유한 차분 사용)
        callback : Callable, optional
            각 반복마다 호출할 콜백 함수
        
        Returns
        -------
        Tuple[np.ndarray, float]
            (최적 위치, 최적 함수값)
        """
        x = np.asarray(x0, dtype=float).copy()
        n = len(x)
        
        self.history = {'x': [], 'f': [], 'grad_norm': [], 'iterations': 0}
        
        for i in range(self.max_iter):
            grad = grad_func(x)
            grad_norm = np.linalg.norm(grad)
            f_val = func(x)
            
            self.history['x'].append(x.copy())
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            
            # 수렴 확인
            if grad_norm < self.tol:
                break
            
            # 헤시안 계산
            if hess_func is not None:
                hess = hess_func(x)
            else:
                hess = self._numerical_hessian(grad_func, x)
            
            # 정규화 (특이 행렬 방지)
            hess_reg = hess + self.damping * np.eye(n)
            
            try:
                # 뉴턴 스텝 계산: H^{-1} * ∇f
                delta = np.linalg.solve(hess_reg, -grad)
            except np.linalg.LinAlgError:
                # 특이 행렬인 경우 경사 하강법 사용
                delta = -grad * 0.01
            
            # 스텝 크기 제한 (안정성)
            step_norm = np.linalg.norm(delta)
            if step_norm > 1.0:
                delta = delta / step_norm
            
            # 위치 업데이트
            x = x + delta
            
            if callback is not None:
                callback(i, x, f_val, grad_norm)
        
        self.history['iterations'] = i + 1
        
        return x, func(x)
    
    def get_history(self) -> dict:
        """최적화 이력 반환"""
        return self.history
