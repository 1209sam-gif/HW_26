"""
다봉 함수 (Multimodal Functions)

다봉 함수는 여러 개의 지역 최솟값을 가지며, 확률론적 최적화 방법의 전역 탐색 능력을 테스트합니다.
"""

import numpy as np


def rastrigin(x):
    """
    Rastrigin 함수
    
    특징:
    - 매우 많은 지역 최솟값 (규칙적인 격자 형태)
    - 전역 최솟값: f(0, 0, ..., 0) = 0
    - 탐색 범위: [-5.12, 5.12]
    - 확률론적 방법 테스트에 적합
    
    수식: f(x) = 10n + Σ[x_i^2 - 10*cos(2πx_i)]
    
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
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def rastrigin_gradient(x):
    """
    Rastrigin 함수의 그래디언트
    
    수식: ∇f(x) = 2x + 20π*sin(2πx)
    """
    x = np.asarray(x)
    return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)


def ackley(x):
    """
    Ackley 함수
    
    특징:
    - 넓고 평평한 바깥 영역과 중앙의 급격한 구멍
    - 전역 최솟값: f(0, 0, ..., 0) = 0
    - 탐색 범위: [-5, 5]
    - 지역 탐색 알고리즘이 쉽게 갇힘
    
    수식: f(x) = -20*exp(-0.2*√(Σx_i^2/n)) - exp(Σcos(2πx_i)/n) + 20 + e
    
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
    n = len(x)
    
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    
    return term1 + term2 + 20 + np.e


def ackley_gradient(x):
    """
    Ackley 함수의 그래디언트
    """
    x = np.asarray(x)
    n = len(x)
    
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    
    sqrt_term = np.sqrt(sum_sq / n) if sum_sq > 0 else 1e-10
    
    term1 = 4 * x * np.exp(-0.2 * sqrt_term) / (n * sqrt_term)
    term2 = 2 * np.pi * np.sin(2 * np.pi * x) * np.exp(sum_cos / n) / n
    
    return term1 + term2


def griewank(x):
    """
    Griewank 함수
    
    특징:
    - 넓은 범위에서 많은 지역 최솟값
    - 전역 최솟값: f(0, 0, ..., 0) = 0
    - 탐색 범위: [-600, 600]
    - 차원이 높을수록 쉬워지는 특이한 특성
    
    수식: f(x) = Σ(x_i^2/4000) - Π(cos(x_i/√i)) + 1
    
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
    n = len(x)
    
    sum_term = np.sum(x ** 2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    
    return sum_term - prod_term + 1


def griewank_gradient(x):
    """
    Griewank 함수의 그래디언트
    """
    x = np.asarray(x)
    n = len(x)
    indices = np.arange(1, n + 1)
    sqrt_indices = np.sqrt(indices)
    
    # 합 항의 그래디언트
    grad_sum = x / 2000
    
    # 곱 항의 그래디언트
    cos_terms = np.cos(x / sqrt_indices)
    sin_terms = np.sin(x / sqrt_indices)
    prod_all = np.prod(cos_terms)
    
    grad_prod = np.zeros(n)
    for i in range(n):
        if cos_terms[i] != 0:
            grad_prod[i] = prod_all * sin_terms[i] / (sqrt_indices[i] * cos_terms[i])
    
    return grad_sum + grad_prod


# 함수 정보 딕셔너리
MULTIMODAL_FUNCTIONS = {
    'rastrigin': {
        'func': rastrigin,
        'gradient': rastrigin_gradient,
        'bounds': (-5.12, 5.12),
        'optimum': 0.0,
        'optimum_point': lambda dim: np.zeros(dim),
        'name_ko': 'Rastrigin 함수',
        'name_en': 'Rastrigin Function'
    },
    'ackley': {
        'func': ackley,
        'gradient': ackley_gradient,
        'bounds': (-5, 5),
        'optimum': 0.0,
        'optimum_point': lambda dim: np.zeros(dim),
        'name_ko': 'Ackley 함수',
        'name_en': 'Ackley Function'
    },
    'griewank': {
        'func': griewank,
        'gradient': griewank_gradient,
        'bounds': (-600, 600),
        'optimum': 0.0,
        'optimum_point': lambda dim: np.zeros(dim),
        'name_ko': 'Griewank 함수',
        'name_en': 'Griewank Function'
    }
}
