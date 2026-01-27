"""
볼록 함수 (Convex Functions)

볼록 함수는 단일 전역 최솟값을 가지며, 결정론적 최적화 방법의 성능을 테스트하는 데 적합합니다.
"""

import numpy as np


def sphere(x):
    """
    Sphere 함수
    
    특징:
    - 가장 단순한 볼록 함수
    - 전역 최솟값: f(0, 0, ..., 0) = 0
    - 탐색 범위: [-5.12, 5.12]
    
    수식: f(x) = Σ(x_i^2)
    
    Parameters
    ----------
    x : array_like
        입력 벡터
    
    Returns
    -------
    float
        함수값
    """
    x = np.asarray(x)
    return np.sum(x ** 2)


def sphere_gradient(x):
    """
    Sphere 함수의 그래디언트
    
    수식: ∇f(x) = 2 * x
    """
    x = np.asarray(x)
    return 2 * x


def rosenbrock(x):
    """
    Rosenbrock 함수 (Banana 함수)
    
    특징:
    - 좁고 포물선 모양의 골짜기
    - 전역 최솟값: f(1, 1, ..., 1) = 0
    - 탐색 범위: [-5, 10]
    - 수렴이 어려운 대표적인 함수
    
    수식: f(x) = Σ[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    
    Parameters
    ----------
    x : array_like
        입력 벡터 (최소 2차원)
    
    Returns
    -------
    float
        함수값
    """
    x = np.asarray(x)
    total = 0.0
    for i in range(len(x) - 1):
        total += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return total


def rosenbrock_gradient(x):
    """
    Rosenbrock 함수의 그래디언트
    """
    x = np.asarray(x)
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n - 1):
        grad[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
        grad[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
    
    return grad


# 함수 정보 딕셔너리
CONVEX_FUNCTIONS = {
    'sphere': {
        'func': sphere,
        'gradient': sphere_gradient,
        'bounds': (-5.12, 5.12),
        'optimum': 0.0,
        'optimum_point': lambda dim: np.zeros(dim),
        'name_ko': 'Sphere 함수',
        'name_en': 'Sphere Function'
    },
    'rosenbrock': {
        'func': rosenbrock,
        'gradient': rosenbrock_gradient,
        'bounds': (-5, 10),
        'optimum': 0.0,
        'optimum_point': lambda dim: np.ones(dim),
        'name_ko': 'Rosenbrock 함수',
        'name_en': 'Rosenbrock Function'
    }
}
