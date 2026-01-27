"""
시각화 도구

최적화 과정과 결과를 시각화하는 함수들을 제공합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, List, Optional, Tuple
import os


def plot_function_3d(
    func: Callable,
    bounds: Tuple[float, float],
    title: str = 'Function Surface',
    resolution: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    3D 표면 플롯을 생성합니다.
    
    Parameters
    ----------
    func : Callable
        2차원 함수 f([x, y])
    bounds : Tuple[float, float]
        탐색 범위 (최소, 최대)
    title : str
        그래프 제목
    resolution : int
        해상도 (격자 수)
    save_path : str, optional
        저장 경로 (없으면 화면에 표시)
    figsize : Tuple[int, int]
        그래프 크기
    """
    low, high = bounds
    x = np.linspace(low, high, resolution)
    y = np.linspace(low, high, resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(
        X, Y, Z,
        cmap='viridis',
        alpha=0.8,
        edgecolor='none'
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title(title)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_contour_with_path(
    func: Callable,
    bounds: Tuple[float, float],
    path: Optional[List[np.ndarray]] = None,
    optimum: Optional[np.ndarray] = None,
    title: str = 'Optimization Path',
    resolution: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    등고선 플롯과 최적화 경로를 시각화합니다.
    
    Parameters
    ----------
    func : Callable
        2차원 함수 f([x, y])
    bounds : Tuple[float, float]
        탐색 범위
    path : List[np.ndarray], optional
        최적화 경로 (위치 리스트)
    optimum : np.ndarray, optional
        실제 최적점
    title : str
        그래프 제목
    resolution : int
        해상도
    save_path : str, optional
        저장 경로
    figsize : Tuple[int, int]
        그래프 크기
    """
    low, high = bounds
    x = np.linspace(low, high, resolution)
    y = np.linspace(low, high, resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 등고선 플롯
    levels = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max()), 30)
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    
    # 최적화 경로
    if path is not None and len(path) > 0:
        path_array = np.array(path)
        ax.plot(
            path_array[:, 0], path_array[:, 1],
            'r.-', linewidth=1.5, markersize=4,
            label='최적화 경로'
        )
        ax.plot(
            path_array[0, 0], path_array[0, 1],
            'go', markersize=10, label='시작점'
        )
        ax.plot(
            path_array[-1, 0], path_array[-1, 1],
            'r*', markersize=15, label='종료점'
        )
    
    # 실제 최적점
    if optimum is not None:
        ax.plot(
            optimum[0], optimum[1],
            'k*', markersize=20, label='전역 최적점'
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    
    plt.colorbar(contour, ax=ax, label='f(X, Y)')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_convergence(
    histories: dict,
    title: str = 'Convergence Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    log_scale: bool = True
) -> None:
    """
    수렴 곡선을 비교합니다.
    
    Parameters
    ----------
    histories : dict
        {알고리즘 이름: 함수값 리스트} 형태의 딕셔너리
    title : str
        그래프 제목
    save_path : str, optional
        저장 경로
    figsize : Tuple[int, int]
        그래프 크기
    log_scale : bool
        로그 스케일 사용 여부
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, f_values in histories.items():
        ax.plot(f_values, label=name, linewidth=2)
    
    ax.set_xlabel('반복 횟수')
    ax.set_ylabel('함수값')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison(
    results: dict,
    metric: str = 'final_value',
    title: str = 'Algorithm Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    알고리즘 성능을 막대 그래프로 비교합니다.
    
    Parameters
    ----------
    results : dict
        {알고리즘 이름: {지표: 값}} 형태의 딕셔너리
    metric : str
        비교할 지표
    title : str
        그래프 제목
    save_path : str, optional
        저장 경로
    figsize : Tuple[int, int]
        그래프 크기
    """
    names = list(results.keys())
    values = [results[name][metric] for name in names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor='black')
    
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # 값 표시
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{value:.4f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom'
        )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
