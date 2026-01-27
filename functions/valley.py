"""
골짜기/고원 함수 (Valley/Plateau Functions)

골짜기 함수는 좁은 골짜기나 평평한 영역을 가지며, 그래디언트 기반 방법의 수렴 특성을 테스트합니다.
"""

import numpy as np


def beale(x):
    """
    Beale 함수
    
    특징:
    - 날카롭고 좁은 골짜기
    - 전역 최솟값: f(3, 0.5) = 0
    - 탐색 범위: [-4.5, 4.5]
    - 2차원 함수
    
    수식: f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    
    Parameters
    ----------
    x : array_like
        입력 벡터 (2차원)
    
    Returns
    -------
    float
        함수값
    """
    x = np.asarray(x)
    x1, x2 = x[0], x[1]
    
    term1 = (1.5 - x1 + x1 * x2) ** 2
    term2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
    term3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
    
    return term1 + term2 + term3


def beale_gradient(x):
    """
    Beale 함수의 그래디언트
    """
    x = np.asarray(x)
    x1, x2 = x[0], x[1]
    
    # 편미분 계산
    f1 = 1.5 - x1 + x1 * x2
    f2 = 2.25 - x1 + x1 * x2 ** 2
    f3 = 2.625 - x1 + x1 * x2 ** 3
    
    df1_dx1 = -1 + x2
    df1_dx2 = x1
    df2_dx1 = -1 + x2 ** 2
    df2_dx2 = 2 * x1 * x2
    df3_dx1 = -1 + x2 ** 3
    df3_dx2 = 3 * x1 * x2 ** 2
    
    grad_x1 = 2 * f1 * df1_dx1 + 2 * f2 * df2_dx1 + 2 * f3 * df3_dx1
    grad_x2 = 2 * f1 * df1_dx2 + 2 * f2 * df2_dx2 + 2 * f3 * df3_dx2
    
    return np.array([grad_x1, grad_x2])


def booth(x):
    """
    Booth 함수
    
    특징:
    - 완만한 골짜기
    - 전역 최솟값: f(1, 3) = 0
    - 탐색 범위: [-10, 10]
    - 2차원 함수
    - 비교적 쉬운 최적화 문제
    
    수식: f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    
    Parameters
    ----------
    x : array_like
        입력 벡터 (2차원)
    
    Returns
    -------
    float
        함수값
    """
    x = np.asarray(x)
    x1, x2 = x[0], x[1]
    
    term1 = (x1 + 2 * x2 - 7) ** 2
    term2 = (2 * x1 + x2 - 5) ** 2
    
    return term1 + term2


def booth_gradient(x):
    """
    Booth 함수의 그래디언트
    """
    x = np.asarray(x)
    x1, x2 = x[0], x[1]
    
    f1 = x1 + 2 * x2 - 7
    f2 = 2 * x1 + x2 - 5
    
    grad_x1 = 2 * f1 + 4 * f2
    grad_x2 = 4 * f1 + 2 * f2
    
    return np.array([grad_x1, grad_x2])


def himmelblau(x):
    """
    Himmelblau 함수
    
    특징:
    - 4개의 동일한 전역 최솟값을 가짐
    - 전역 최솟값들: f(3, 2), f(-2.805, 3.131), f(-3.779, -3.283), f(3.584, -1.848) = 0
    - 탐색 범위: [-5, 5]
    - 2차원 함수
    
    수식: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    
    Parameters
    ----------
    x : array_like
        입력 벡터 (2차원)
    
    Returns
    -------
    float
        함수값
    """
    x = np.asarray(x)
    x1, x2 = x[0], x[1]
    
    term1 = (x1 ** 2 + x2 - 11) ** 2
    term2 = (x1 + x2 ** 2 - 7) ** 2
    
    return term1 + term2


def himmelblau_gradient(x):
    """
    Himmelblau 함수의 그래디언트
    """
    x = np.asarray(x)
    x1, x2 = x[0], x[1]
    
    f1 = x1 ** 2 + x2 - 11
    f2 = x1 + x2 ** 2 - 7
    
    grad_x1 = 4 * x1 * f1 + 2 * f2
    grad_x2 = 2 * f1 + 4 * x2 * f2
    
    return np.array([grad_x1, grad_x2])


# 함수 정보 딕셔너리
VALLEY_FUNCTIONS = {
    'beale': {
        'func': beale,
        'gradient': beale_gradient,
        'bounds': (-4.5, 4.5),
        'optimum': 0.0,
        'optimum_point': lambda dim: np.array([3.0, 0.5]),
        'dim': 2,
        'name_ko': 'Beale 함수',
        'name_en': 'Beale Function'
    },
    'booth': {
        'func': booth,
        'gradient': booth_gradient,
        'bounds': (-10, 10),
        'optimum': 0.0,
        'optimum_point': lambda dim: np.array([1.0, 3.0]),
        'dim': 2,
        'name_ko': 'Booth 함수',
        'name_en': 'Booth Function'
    },
    'himmelblau': {
        'func': himmelblau,
        'gradient': himmelblau_gradient,
        'bounds': (-5, 5),
        'optimum': 0.0,
        'optimum_point': lambda dim: np.array([3.0, 2.0]),  # 4개 중 하나
        'dim': 2,
        'name_ko': 'Himmelblau 함수',
        'name_en': 'Himmelblau Function'
    }
}
