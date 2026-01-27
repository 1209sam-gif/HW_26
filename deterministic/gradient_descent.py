"""
경사 하강법 (Gradient Descent)

가장 기본적인 최적화 알고리즘으로, 그래디언트의 반대 방향으로 이동하여 최솟값을 찾습니다.
"""

import numpy as np
from typing import Callable, Tuple, List, Optional


class GradientDescent:
    """
    경사 하강법 최적화기
    
    여러 가지 변형을 지원합니다:
    - 고정 학습률 (Fixed learning rate)
    - 모멘텀 (Momentum)
    - 선탐색 (Line search)
    
    Attributes
    ----------
    learning_rate : float
        학습률 (스텝 크기)
    momentum : float
        모멘텀 계수 (0이면 사용 안함)
    max_iter : int
        최대 반복 횟수
    tol : float
        수렴 허용 오차
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        use_line_search: bool = False
    ):
        """
        Parameters
        ----------
        learning_rate : float
            학습률 (기본값: 0.01)
        momentum : float
            모멘텀 계수 (기본값: 0.0, 사용 안함)
        max_iter : int
            최대 반복 횟수 (기본값: 1000)
        tol : float
            수렴 허용 오차 (기본값: 1e-6)
        use_line_search : bool
            선탐색 사용 여부 (기본값: False)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        self.use_line_search = use_line_search
        
        # 최적화 이력
        self.history = {
            'x': [],           # 위치 이력
            'f': [],           # 함수값 이력
            'grad_norm': [],   # 그래디언트 노름 이력
            'iterations': 0    # 반복 횟수
        }
    
    def _line_search(
        self,
        func: Callable,
        x: np.ndarray,
        direction: np.ndarray,
        c1: float = 1e-4,
        c2: float = 0.9
    ) -> float:
        """
        Armijo 조건을 사용한 백트래킹 선탐색
        
        Parameters
        ----------
        func : Callable
            목적 함수
        x : np.ndarray
            현재 위치
        direction : np.ndarray
            탐색 방향 (음의 그래디언트)
        c1 : float
            Armijo 조건 상수
        c2 : float
            축소 비율
        
        Returns
        -------
        float
            최적 스텝 크기
        """
        alpha = 1.0
        f_x = func(x)
        
        while alpha > 1e-10:
            x_new = x + alpha * direction
            f_new = func(x_new)
            
            # Armijo 조건: f(x + αd) ≤ f(x) + c1 * α * ∇f(x)^T * d
            if f_new <= f_x + c1 * alpha * np.dot(-direction, direction):
                return alpha
            
            alpha *= c2
        
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
        velocity = np.zeros_like(x)
        
        # 이력 초기화
        self.history = {'x': [], 'f': [], 'grad_norm': [], 'iterations': 0}
        
        for i in range(self.max_iter):
            # 그래디언트 계산
            grad = grad_func(x)
            grad_norm = np.linalg.norm(grad)
            f_val = func(x)
            
            # 이력 저장
            self.history['x'].append(x.copy())
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            
            # 수렴 확인
            if grad_norm < self.tol:
                break
            
            # 탐색 방향 계산 (음의 그래디언트)
            direction = -grad
            
            # 스텝 크기 결정
            if self.use_line_search:
                lr = self._line_search(func, x, direction)
            else:
                lr = self.learning_rate
            
            # 모멘텀 적용
            velocity = self.momentum * velocity + lr * direction
            
            # 위치 업데이트
            x = x + velocity
            
            # 콜백 호출
            if callback is not None:
                callback(i, x, f_val, grad_norm)
        
        self.history['iterations'] = i + 1
        
        return x, func(x)
    
    def get_history(self) -> dict:
        """
        최적화 이력 반환
        
        Returns
        -------
        dict
            최적화 과정의 이력
        """
        return self.history
