"""
성능 평가 지표

최적화 알고리즘의 성능을 측정하고 비교하는 도구를 제공합니다.
"""

import numpy as np
import time
from typing import Callable, Tuple, Dict, Any, Optional
import pandas as pd


class OptimizationMetrics:
    """
    최적화 성능 평가 도구
    
    다양한 지표를 측정하고 결과를 정리합니다.
    """
    
    def __init__(self):
        """초기화"""
        self.results = []
    
    def evaluate(
        self,
        optimizer,
        func: Callable,
        grad_func: Optional[Callable],
        x0_or_dim: Any,
        bounds: Tuple[float, float],
        true_optimum: float,
        true_optimum_point: np.ndarray,
        optimizer_name: str,
        function_name: str,
        n_runs: int = 1,
        **kwargs
    ) -> Dict[str, float]:
        """
        최적화 알고리즘 평가
        
        Parameters
        ----------
        optimizer : object
            최적화기 객체
        func : Callable
            목적 함수
        grad_func : Callable, optional
            그래디언트 함수 (결정론적 방법에서 사용)
        x0_or_dim : Any
            초기점 또는 차원
        bounds : Tuple[float, float]
            탐색 범위
        true_optimum : float
            실제 최적값
        true_optimum_point : np.ndarray
            실제 최적점
        optimizer_name : str
            최적화기 이름
        function_name : str
            함수 이름
        n_runs : int
            실행 횟수 (확률론적 방법의 경우)
        **kwargs : dict
            추가 인자
        
        Returns
        -------
        Dict[str, float]
            평가 결과
        """
        final_values = []
        final_errors = []
        point_errors = []
        times = []
        iterations = []
        
        for _ in range(n_runs):
            start_time = time.time()
            
            # 결정론적 방법과 확률론적 방법 구분
            if hasattr(optimizer, 'optimize') and grad_func is not None:
                # 결정론적 방법
                if isinstance(x0_or_dim, int):
                    x0 = np.random.uniform(bounds[0], bounds[1], x0_or_dim)
                else:
                    x0 = x0_or_dim
                best_x, best_f = optimizer.optimize(func, grad_func, x0, **kwargs)
            else:
                # 확률론적 방법
                dim = x0_or_dim if isinstance(x0_or_dim, int) else len(x0_or_dim)
                best_x, best_f = optimizer.optimize(func, dim, bounds, **kwargs)
            
            elapsed_time = time.time() - start_time
            
            # 지표 계산
            final_values.append(best_f)
            final_errors.append(abs(best_f - true_optimum))
            point_errors.append(np.linalg.norm(best_x - true_optimum_point))
            times.append(elapsed_time)
            
            history = optimizer.get_history()
            if 'iterations' in history:
                iterations.append(history['iterations'])
            elif 'generations' in history:
                iterations.append(history['generations'])
        
        # 결과 집계
        result = {
            'optimizer': optimizer_name,
            'function': function_name,
            'final_value': np.mean(final_values),
            'final_value_std': np.std(final_values),
            'final_error': np.mean(final_errors),
            'point_error': np.mean(point_errors),
            'time': np.mean(times),
            'iterations': np.mean(iterations) if iterations else 0,
            'success_rate': np.mean([e < 1e-3 for e in final_errors])
        }
        
        self.results.append(result)
        
        return result
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        결과를 DataFrame으로 반환
        
        Returns
        -------
        pd.DataFrame
            모든 평가 결과
        """
        return pd.DataFrame(self.results)
    
    def save_results(self, filepath: str) -> None:
        """
        결과를 CSV 파일로 저장
        
        Parameters
        ----------
        filepath : str
            저장 경로
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = self.get_results_dataframe()
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    def summary(self) -> pd.DataFrame:
        """
        최적화기별 요약 통계
        
        Returns
        -------
        pd.DataFrame
            요약 통계
        """
        df = self.get_results_dataframe()
        summary = df.groupby('optimizer').agg({
            'final_error': ['mean', 'std'],
            'time': ['mean', 'std'],
            'iterations': 'mean',
            'success_rate': 'mean'
        })
        return summary
    
    def clear(self) -> None:
        """결과 초기화"""
        self.results = []
