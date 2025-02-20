import time
import multiprocessing
import torch

def cpu_bound_task(n):
    """Fonction CPU-bound pour effectuer des calculs intensifs."""
    total = 0
    for i in range(n):
        total += i * i
    return total

def measure_cpu_performance():
    """Mesure la performance CPU en utilisant des jobs en parallèle."""
    num_workers = multiprocessing.cpu_count()
    n = 10**6
    start_time = time.time()
    
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(cpu_bound_task, [n] * num_workers)
    
    end_time = time.time()
    print(f"Temps d'exécution CPU avec {num_workers} workers: {end_time - start_time:.2f} secondes")

def measure_gpu_performance():
    """Mesure la performance GPU en utilisant PyTorch."""
    if not torch.cuda.is_available():
        print("Aucun GPU disponible.")
        return
    
    device = torch.device('cuda')
    start_time = time.time()
    
    a = torch.randn(10000, 10000, device=device)
    b = torch.randn(10000, 10000, device=device)
    c = torch.matmul(a, b)
    
    end_time = time.time()
    print(f"Temps d'exécution GPU: {end_time - start_time:.2f} secondes")

if __name__ == "__main__":
    print("Mesure de la performance CPU:")
    measure_cpu_performance()
    
    print("\nMesure de la performance GPU:")
    measure_gpu_performance()