import random
import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.setrecursionlimit(2000000)

# ==========================================
# PHẦN 1: CÁC THUẬT TOÁN SẮP XẾP
# ==========================================
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)


def numpy_sort(arr):
    return np.sort(arr)


# ==========================================
# PHẦN 2: QUẢN LÝ DỮ LIỆU (SINH & LƯU FILE)
# ==========================================
DATA_FILE = "dataset.csv"
NMAX = 100000000

def generate_and_save_data(size=1000000):
    print(f"Đang sinh bộ dữ liệu mới ({size} phần tử/dãy)...")

    # 1. Tạo cấu hình ngẫu nhiên (5 float, 5 int)
    data_types = ['float'] * 5 + ['int'] * 5
    random.shuffle(data_types)

    dataset_dict = {}

    print(f"Thứ tự sinh ngẫu nhiên: {data_types}")

    for i, dtype in enumerate(data_types):

        if dtype == 'float':
            data = np.random.uniform(-NMAX, NMAX, size)
            type_label = "Float"
        else:
            data = np.random.randint(-NMAX, NMAX, size)
            type_label = "Int"


        col_name = ""
        if i == 0:
            data = np.sort(data)
            col_name = f"Seq{i + 1}_{type_label}_SortedAsc"
        elif i == 1:
            data = np.sort(data)[::-1]
            col_name = f"Seq{i + 1}_{type_label}_SortedDesc"
        else:
            col_name = f"Seq{i + 1}_{type_label}_Random"

        dataset_dict[col_name] = data

    # 2. Lưu ra file CSV
    print("Đang lưu dữ liệu ra file dataset.csv (có thể mất vài giây)...")
    df = pd.DataFrame(dataset_dict)
    df.to_csv(DATA_FILE, index=False)
    print("Đã lưu xong!")

    # Trả về format cũ để chạy benchmark
    return [df[col].tolist() for col in df.columns], df.columns.tolist()


def load_data_from_file():
    print(f"Phát hiện file {DATA_FILE}. Đang đọc dữ liệu lên...")
    df = pd.read_csv(DATA_FILE)
    print("Đã đọc xong dữ liệu cũ.")

    datasets = []
    for col in df.columns:
        datasets.append(df[col].tolist())

    return datasets, df.columns.tolist()


# ==========================================
# PHẦN 3: THỰC THI VÀ ĐO THỜI GIAN
# ==========================================
def benchmark():

    SIZE = 1000000

    if os.path.exists(DATA_FILE):
        user_choice = input("Đã có file 'dataset.csv'. Bạn muốn dùng lại (y) hay tạo mới (n)? ")
        if user_choice.lower() == 'y':
            datasets, names = load_data_from_file()
        else:
            datasets, names = generate_and_save_data(SIZE)
    else:
        datasets, names = generate_and_save_data(SIZE)

    results = {
        'Dataset': names,
        'QuickSort': [],
        'HeapSort': [],
        'MergeSort': [],
        'NumPy Sort': []
    }

    algorithms = [
        ('QuickSort', quick_sort),
        ('HeapSort', heap_sort),
        ('MergeSort', merge_sort),
        ('NumPy Sort', numpy_sort)
    ]

    print("\n" + "=" * 50)
    print("BẮT ĐẦU BENCHMARK")
    print("=" * 50)

    for algo_name, algo_func in algorithms:
        print(f"\n>>> Chạy {algo_name}...")
        times = []
        for i, data in enumerate(datasets):

            arr_copy = data.copy()
            if algo_name == 'NumPy Sort':
                arr_input = np.array(arr_copy)
            else:
                arr_input = arr_copy

            start_time = time.perf_counter()
            algo_func(arr_input)
            end_time = time.perf_counter()

            exec_time_ms = (end_time - start_time) * 1000
            exec_time_ms = round(exec_time_ms, 3)

            times.append(exec_time_ms)
            print(f"   - {names[i]}: {exec_time_ms}s")

        results[algo_name] = times

    # ==========================================
    # PHẦN 4: LƯU BÁO CÁO
    # ==========================================
    # Lưu kết quả đo thời gian
    df_result = pd.DataFrame(results)
    mean_values = df_result.iloc[:, 1:].mean().round(2)

    avg_row = pd.DataFrame([['AVERAGE'] + mean_values.tolist()], columns=df_result.columns)
    df_final = pd.concat([df_result, avg_row], ignore_index=True)

    df_final.to_csv('sorting_benchmark_result.csv', index=False)

    # Vẽ biểu đồ
    plt.figure(figsize=(14, 8))
    x = np.arange(len(names))
    width = 0.2

    plt.bar(x - 1.5 * width, df_result['QuickSort'], width, label='QuickSort')
    plt.bar(x - 0.5 * width, df_result['HeapSort'], width, label='HeapSort')
    plt.bar(x + 0.5 * width, df_result['MergeSort'], width, label='MergeSort')
    plt.bar(x + 1.5 * width, df_result['NumPy Sort'], width, label='NumPy Sort')

    plt.xlabel('Bộ dữ liệu')
    plt.ylabel('Thời gian (ms)')
    plt.title(f'Benchmark Thuật toán Sắp xếp (Size={len(datasets[0])})')
    plt.xticks(x, names, rotation=45, ha='right', fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sorting_chart.png')

    print("\n" + "=" * 50)
    print("HOÀN TẤT!")
    print("1. Dữ liệu thô: dataset.csv")
    print("2. Bảng kết quả chạy: sorting_benchmark_result.csv")
    print("3. Biểu đồ: sorting_chart.png")
    plt.show()


if __name__ == "__main__":
    benchmark()

