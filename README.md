# High-Performance k-NN Classification Pipeline in C++

This project implements an optimized **K-Nearest Neighbors (k-NN)** classification pipeline in **C++**, supporting both **brute-force** and **KD-Tree**–based search strategies.  
It is designed for **high-dimensional datasets** (e.g., Fashion MNIST with 784 features) and focuses on **modularity, performance, and controlled experimentation**.

---

## Design Rationale

The project prioritizes **clarity, modularity, and performance** over external dependencies.

Key design decisions:
- A lightweight vector abstraction to explicitly control numerical operations
- Multiple k-NN backends (brute-force and KD-Tree) to study algorithmic trade-offs
- OpenMP-based parallelism to evaluate scalability during inference
- Minimal dependencies to keep performance characteristics transparent

> **Note:** While KD-Trees can degrade in very high-dimensional spaces, they are included here to study practical trade-offs between theoretical complexity and real-world performance.

---

## Features

- Modular `Vector` class for numerical computations
- Custom k-NN classifier with interchangeable backends
- OpenMP-enabled parallel inference (up to ~55× speedup)
- CLI-based benchmarking and evaluation
- CSV dataset loading and synthetic data generation
- Designed with extensibility and experimentation in mind

---

## Project Structure

```

.
├── include/
│   ├── cli_options.hpp         # CLI argument parsing
│   ├── cli_parser.hpp          # Custom CLI parser
│   ├── data_utils.hpp          # CSV loading and preprocessing
│   ├── evaluate.hpp            # Evaluation utilities
│   ├── kdtree_knn.hpp          # KD-Tree-based k-NN
│   ├── kdtree.hpp              # KD-Tree data structure
│   ├── knn.hpp                 # Brute-force k-NN implementation
│   └── vector.hpp              # Vector math abstraction
├── src/
│   ├── knn_brute_force_main.cpp             # Brute-force pipeline
│   ├── knn_cli_main.cpp        # CLI benchmarking tool
│   └── knn_kdtree_main.cpp     # KD-Tree pipeline
├── datasets/
│   └── fashion_mnist_10k.csv   # Sample dataset (10k samples, 784 features)
└── README.md

````

---

## Module Overview

### Vector Abstraction (`include/vector.hpp`)
- Wrapper over `std::vector<double>`
- Supports dot product, L2 norm, and squared Euclidean distance
- Exception-safe and performance-conscious
- Used consistently across all components

---

### Brute-Force k-NN (`include/knn.hpp`)
- Naive k-NN using squared Euclidean distance
- Efficient neighbor selection via `std::partial_sort`
- Parallel prediction using OpenMP
- Deterministic majority voting with tie-breaking

---

### Data Utilities (`include/data_utils.hpp`)
- CSV parsing with malformed row handling
- Synthetic data generation for controlled testing
- Train-test split with fixed random seed for reproducibility

---

### Evaluation (`include/evaluate.hpp`)
- Accuracy computation
- Input validation and consistency checks

---

### Brute-Force Pipeline (`src/knn_main.cpp`)
- End-to-end training and evaluation on Fashion MNIST
- Results:
  - Inference time reduced from ~362s to ~6.5s using OpenMP (~55× speedup)
  - Accuracy: 84.04%
- Uses `std::chrono` for performance measurement

---

### KD-Tree Accelerator (`include/kdtree.hpp`)
- Recursive KD-Tree with median-based splits
- Max-heap–based top-k querying with pruning
- Designed for efficient nearest-neighbor search experimentation

---

### KD-Tree k-NN (`include/kdtree_knn.hpp`)
- Modular KD-Tree-backed k-NN classifier
- OpenMP-enabled inference
- Safe memory management for scalability

---

### KD-Tree Evaluation (`src/kdtree_knn_main.cpp`)
- Benchmarks KD-Tree inference on Fashion MNIST
- Results:
  - Inference time: ~19s → ~7.5s (≈2.5× speedup with parallelism)
  - Accuracy unchanged at 84.04%

---

### CLI Benchmarking Tool (`src/knn_cli_main.cpp`)
- Built using `cxxopts`
- Configurable parameters:
  - Dataset path
  - Train-test split ratio
  - Number of neighbors
  - Backend selection (brute-force / KD-Tree)
  - Parallel execution toggle
- Outputs inference time and classification accuracy

---

##  Dependencies

- C++17 or later
- OpenMP (for parallelism)
- [cxxopts](https://github.com/jarro2783/cxxopts) for CLI argument parsing

---

## Performance Summary (Fashion MNIST)

| Method      | Parallel | Inference Time | Accuracy(k=12) |
|:-----------:|:--------:|:--------------:|:--------:|
| Brute-Force | ❌       | ~362s          | 84.04%   |
| Brute-Force | ✅       | ~6.5s          | 84.04%   |
| KD-Tree     | ❌       | ~19s           | 84.04%   |
| KD-Tree     | ✅       | ~7.5s          | 84.04%   |

---

## Sample CLI Usage

````bash
./knn_cli_main \
  --dataset "datasets/fashion_mnist_10k.csv" \
  --split 0.8 \
  --neighbors 5 \
  --mode kdtree \
  --parallel true 
````

---

##  Future Work

* Weighted k-NN variants
* Dimensionality reduction (e.g., PCA) for preprocessing
* Multi-label classification support
* Profiling and cache-level performance analysis

---

##  License

[MIT License](LICENSE)

---

## Acknowledgments

* [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
* [cxxopts CLI Parser](https://github.com/jarro2783/cxxopts)