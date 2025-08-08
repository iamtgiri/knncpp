# 🚀 High-Performance k-NN Classification Pipeline in C++

This project implements an optimized **K-Nearest Neighbors (k-NN)** classification system in **C++**, supporting both **brute-force** and **KD-Tree** search strategies. It’s designed for high-dimensional datasets (e.g., Fashion MNIST with 734 features) and focuses on modularity, performance, and extensibility.

---

## 🔧 Features

- Modular `Vector` class abstraction for numerical operations
- Custom k-NN classifier with brute-force and KD-Tree backends
- OpenMP-enabled parallel inference (up to 55× speedup)
- CLI benchmarking tool for performance testing
- Supports CSV dataset loading, synthetic data generation, and evaluation
- Designed for large-scale classification tasks in resource-constrained environments

---

## 📁 Project Structure

```
.
├── include/
│   ├── vector.hpp              # Custom vector math library
│   ├── knn.hpp                 # Brute-force KNN classifier
│   ├── kdtree.hpp              # KD-Tree data structure
│   ├── kdtree\_knn.hpp          # KD-Tree-based KNN classifier
│   ├── data\_utils.hpp          # CSV loader and data preprocessing
│   ├── evaluate.hpp             # Evaluation utilities
├── src/
│   ├── knn\_main.cpp             # Brute-force KNN application
│   ├── kdtree\_knn\_main.cpp     # KD-Tree KNN application
│   ├── knn\_cli\_main.cpp        # CLI benchmarking tool
├── datasets/
│   ├── fashion\_mnist\_10k.csv   # Sample dataset (10k samples, 734 features)
├── README.md

```

---

## 📌 Modules Overview

### 🔹 Custom Vector Class (`include/vector.hpp`)
- Abstraction over `std::vector<double>` for numerical ops
- Supports dot product, L2 norm, squared Euclidean distance
- Exception-safe, optimized for performance
- Used across all classifiers to maintain consistent math operations

---

### 🔹 Brute-Force KNN Classifier (`include/knn.hpp`)
- Naive k-NN with squared Euclidean distance
- Efficient neighbor retrieval via `std::partial_sort`
- Parallel prediction using **OpenMP**
- Deterministic majority voting with tie-breaking

---

### 🔹 Data Utilities (`include/data_utils.hpp`)
- CSV parser (handles malformed rows)
- Synthetic data generation for binary classification
- Train-test split with reproducibility using fixed random seed

---

### 🔹 Evaluation (`include/evaluate.hpp`)
- Accuracy score computation
- Input validation and error checking

---

### 🔹 Main Pipeline - Brute Force (`src/knn_main.cpp`)
- Trains and evaluates brute-force KNN on Fashion MNIST
- Achieved:
  - ⏱ ~55× speedup using OpenMP (362s → 6.5s)
  - 🎯 Accuracy: 84.04%
- Includes performance benchmarking via `std::chrono`

---

### 🔹 KD-Tree Accelerator (`include/kdtree.hpp`)
- Recursive KD-Tree with median splits
- Max-heap for top-k querying with pruning
- Designed for fast lookup in high-dimensional space

---

### 🔹 KD-Tree KNN Classifier (`include/kdtree_knn.hpp`)
- Modular KD-Tree KNN with OpenMP support
- Dynamic tree construction and majority voting
- Manages memory safely for scalable integration

---

### 🔹 KD-Tree Evaluation (`src/kdtree_knn_main.cpp`)
- Benchmarks KD-Tree on Fashion MNIST
- Results:
  - 🔄 Inference Time: 19s → 7.5s (2.5× speedup)
  - 🎯 Accuracy: 84.04% (same as brute-force)

---

### 🔹 CLI Benchmarking Tool (`src/knn_cli_main.cpp`)
- Built with `cxxopts` for argument parsing
- Supports:
  - Dataset path
  - Train/test split
  - Number of neighbors
  - Mode: brute-force or KD-Tree
  - Parallel toggle (OpenMP)
- Outputs:
  - ⏱ Prediction time
  - 🎯 Classification accuracy

---

## ⚙️ Dependencies

- C++17 or above
- OpenMP (for parallelism)
- [cxxopts](https://github.com/jarro2783/cxxopts) for CLI parsing

---

## 📊 Performance Summary (Fashion MNIST)

| Method      | Parallel | Inference Time | Accuracy |
|-------------|----------|----------------|----------|
| Brute-Force | ❌       | ~362s          | 84.04%   |
| Brute-Force | ✅       | ~6.5s          | 84.04%   |
| KD-Tree     | ❌       | ~19s           | 84.04%   |
| KD-Tree     | ✅       | ~7.5s          | 84.04%   |

---

## 🧪 Sample CLI Usage

```bash
./knn_cli_main \
  --dataset "datasets/fashion_mnist_10k.csv" \
  --split 0.8 \
  --neighbors 5 \
  --mode kd-tree \
  --parallel true
````

---

## 🧠 Future Work

* Add support for weighted k-NN
* Explore dimensionality reduction (PCA, t-SNE) for preprocessing
* Extend to multi-label classification
* Visualize decision boundaries in 2D embeddings

---

## 📄 License

This project is released under the MIT License.

---

## 🙌 Acknowledgments

* [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
* [cxxopts CLI Parser](https://github.com/jarro2783/cxxopts)


