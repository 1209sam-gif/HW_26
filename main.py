"""
최적화 학습 프로젝트 - 메인 실행 파일

이 스크립트는 다양한 최적화 알고리즘을 벤치마크 함수에 대해 테스트하고 비교합니다.
"""

import numpy as np
import os

# 벤치마크 함수
from functions.convex import sphere, sphere_gradient, rosenbrock, rosenbrock_gradient, CONVEX_FUNCTIONS
from functions.multimodal import rastrigin, rastrigin_gradient, ackley, ackley_gradient, MULTIMODAL_FUNCTIONS
from functions.valley import beale, beale_gradient, booth, booth_gradient, VALLEY_FUNCTIONS

# 결정론적 최적화 방법
from deterministic import GradientDescent, NewtonMethod, BFGS

# 확률론적 최적화 방법
from stochastic import (
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
    DifferentialEvolution
)

# 유틸리티
from utils import (
    plot_function_3d,
    plot_contour_with_path,
    plot_convergence,
    plot_comparison,
    OptimizationMetrics
)


def test_deterministic_methods():
    """결정론적 최적화 방법 테스트"""
    print("=" * 60)
    print("결정론적 최적화 방법 테스트")
    print("=" * 60)
    
    # Rosenbrock 함수로 테스트
    func = rosenbrock
    grad = rosenbrock_gradient
    x0 = np.array([-1.5, 2.0])
    
    optimizers = [
        ("경사 하강법", GradientDescent(learning_rate=0.001, max_iter=5000)),
        ("경사 하강법 (모멘텀)", GradientDescent(learning_rate=0.001, momentum=0.9, max_iter=5000)),
        ("뉴턴 방법", NewtonMethod(max_iter=100)),
        ("BFGS", BFGS(max_iter=500))
    ]
    
    histories = {}
    
    for name, optimizer in optimizers:
        best_x, best_f = optimizer.optimize(func, grad, x0.copy())
        history = optimizer.get_history()
        histories[name] = history['f']
        
        print(f"\n{name}:")
        print(f"  최적점: [{best_x[0]:.6f}, {best_x[1]:.6f}]")
        print(f"  최적값: {best_f:.6e}")
        print(f"  반복 횟수: {history['iterations']}")
    
    return histories


def test_stochastic_methods():
    """확률론적 최적화 방법 테스트"""
    print("\n" + "=" * 60)
    print("확률론적 최적화 방법 테스트")
    print("=" * 60)
    
    # Rastrigin 함수로 테스트 (다봉 함수)
    func = rastrigin
    dim = 2
    bounds = (-5.12, 5.12)
    
    np.random.seed(42)
    
    optimizers = [
        ("유전 알고리즘", GeneticAlgorithm(pop_size=50, max_generations=100)),
        ("입자 군집 최적화", ParticleSwarmOptimization(n_particles=30, max_iter=100)),
        ("담금질 기법", SimulatedAnnealing(initial_temp=100, cooling_rate=0.95)),
        ("차분 진화", DifferentialEvolution(pop_size=50, max_generations=100))
    ]
    
    histories = {}
    
    for name, optimizer in optimizers:
        best_x, best_f = optimizer.optimize(func, dim, bounds)
        history = optimizer.get_history()
        
        if 'best_f' in history:
            histories[name] = history['best_f']
        
        print(f"\n{name}:")
        print(f"  최적점: [{best_x[0]:.6f}, {best_x[1]:.6f}]")
        print(f"  최적값: {best_f:.6e}")
    
    return histories


def visualize_functions(output_dir: str = './outputs/figures'):
    """벤치마크 함수 시각화"""
    print("\n" + "=" * 60)
    print("벤치마크 함수 시각화")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 모든 2D 함수 시각화
    functions_2d = [
        ('Sphere', sphere, (-5.12, 5.12)),
        ('Rosenbrock', rosenbrock, (-2, 2)),
        ('Rastrigin', rastrigin, (-5.12, 5.12)),
        ('Ackley', ackley, (-5, 5)),
        ('Beale', beale, (-4.5, 4.5)),
        ('Booth', booth, (-10, 10)),
    ]
    
    for name, func, bounds in functions_2d:
        save_path = os.path.join(output_dir, f'{name.lower()}_3d.png')
        print(f"  {name} 함수 3D 플롯 생성 중...")
        plot_function_3d(func, bounds, title=f'{name} Function', save_path=save_path)
    
    print(f"\n그래프가 {output_dir}에 저장되었습니다.")


def run_full_comparison():
    """전체 비교 실험 수행"""
    print("\n" + "=" * 60)
    print("전체 비교 실험")
    print("=" * 60)
    
    metrics = OptimizationMetrics()
    
    # 테스트할 함수들
    test_functions = [
        ('Sphere', sphere, sphere_gradient, (-5.12, 5.12), 0.0, np.zeros(2)),
        ('Rosenbrock', rosenbrock, rosenbrock_gradient, (-5, 10), 0.0, np.ones(2)),
        ('Rastrigin', rastrigin, rastrigin_gradient, (-5.12, 5.12), 0.0, np.zeros(2)),
    ]
    
    for func_name, func, grad, bounds, opt_val, opt_point in test_functions:
        print(f"\n{func_name} 함수 테스트:")
        
        # 결정론적 방법
        for opt_name, optimizer in [
            ('GradientDescent', GradientDescent(learning_rate=0.01, max_iter=1000)),
            ('BFGS', BFGS(max_iter=500))
        ]:
            result = metrics.evaluate(
                optimizer, func, grad, 2, bounds,
                opt_val, opt_point, opt_name, func_name
            )
            print(f"  {opt_name}: 오차 = {result['final_error']:.6e}, 시간 = {result['time']:.4f}초")
        
        # 확률론적 방법
        for opt_name, optimizer in [
            ('PSO', ParticleSwarmOptimization(n_particles=30, max_iter=100)),
            ('DE', DifferentialEvolution(pop_size=50, max_generations=100))
        ]:
            result = metrics.evaluate(
                optimizer, func, None, 2, bounds,
                opt_val, opt_point, opt_name, func_name
            )
            print(f"  {opt_name}: 오차 = {result['final_error']:.6e}, 시간 = {result['time']:.4f}초")
    
    # 결과 저장
    metrics.save_results('./outputs/results/comparison_results.csv')
    print("\n결과가 './outputs/results/comparison_results.csv'에 저장되었습니다.")
    
    return metrics


def main():
    """메인 함수"""
    print("=" * 60)
    print("최적화 학습 프로젝트")
    print("=" * 60)
    print("\n1. 결정론적 방법 테스트")
    print("2. 확률론적 방법 테스트")
    print("3. 함수 시각화")
    print("4. 전체 비교 실험")
    print("5. 모두 실행")
    print("0. 종료")
    
    choice = input("\n선택하세요 (0-5): ").strip()
    
    if choice == '1':
        test_deterministic_methods()
    elif choice == '2':
        test_stochastic_methods()
    elif choice == '3':
        visualize_functions()
    elif choice == '4':
        run_full_comparison()
    elif choice == '5':
        test_deterministic_methods()
        test_stochastic_methods()
        visualize_functions()
        run_full_comparison()
    elif choice == '0':
        print("프로그램을 종료합니다.")
    else:
        print("잘못된 선택입니다.")


if __name__ == '__main__':
    main()
