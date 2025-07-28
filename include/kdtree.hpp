// kdtree.hpp

#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <stdexcept>

// ==========================
// KDNode Struct
// Represents a single node in the KD-Tree with a feature vector, label, splitting axis, and children
// ==========================
struct KDNode {
    std::vector<double> point;   // Data point / feature vector
    int label;                   // Associated class label
    int axis;                    // Dimension used to split at this node
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode(const std::vector<double>& pt, int lbl, int ax)
        : point(pt), label(lbl), axis(ax), left(nullptr), right(nullptr) {}
};

// ==========================
// KDTree Class
// Implements KD-Tree construction and k-NN query functionality
// ==========================
class KDTree {
private:
    std::unique_ptr<KDNode> root;  // Root of the KD-Tree
    size_t dims;                   // Number of dimensions in input data

    // Recursive function to build the KD-Tree
    std::unique_ptr<KDNode> build(std::vector<std::pair<std::vector<double>, int>>& points, int depth) {
        if (points.empty()) return nullptr;

        int axis = depth % dims;

        // Partition by median on current axis
        auto comparator = [axis](const auto& a, const auto& b) {
            return a.first[axis] < b.first[axis];
        };

        size_t mid = points.size() / 2;
        std::nth_element(points.begin(), points.begin() + mid, points.end(), comparator);

        auto median = points[mid];
        std::vector<std::pair<std::vector<double>, int>> left(points.begin(), points.begin() + mid);
        std::vector<std::pair<std::vector<double>, int>> right(points.begin() + mid + 1, points.end());

        auto node = std::make_unique<KDNode>(median.first, median.second, axis);
        node->left = build(left, depth + 1);
        node->right = build(right, depth + 1);

        return node;
    }

    // Compute squared Euclidean distance between two vectors
    double distance_squared(const std::vector<double>& a, const std::vector<double>& b) const {
        double dist = 0.0;
        for (size_t i = 0; i < dims; ++i)
            dist += (a[i] - b[i]) * (a[i] - b[i]);
        return dist;
    }

    // Recursive k-NN search function
    void knn_search(const KDNode* node, const std::vector<double>& target, int k,
                    std::vector<std::pair<double, int>>& best) const
    {
        if (!node) return;

        double dist = distance_squared(node->point, target);

        if (best.size() < k) {
            best.emplace_back(dist, node->label);
            std::push_heap(best.begin(), best.end(), compare);
        } else if (dist < best.front().first) {
            std::pop_heap(best.begin(), best.end(), compare);
            best.back() = { dist, node->label };
            std::push_heap(best.begin(), best.end(), compare);
        }

        int axis = node->axis;
        bool goLeft = target[axis] < node->point[axis];

        const KDNode* first = goLeft ? node->left.get() : node->right.get();
        const KDNode* second = goLeft ? node->right.get() : node->left.get();

        knn_search(first, target, k, best);

        // Check if other side of the tree could contain closer points
        double diff = target[axis] - node->point[axis];
        if (best.size() < k || diff * diff < best.front().first) {
            knn_search(second, target, k, best);
        }
    }

    // Comparator for maintaining max-heap of top-k closest neighbors
    static bool compare(const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first < b.first;
    }

public:
    // Constructor: build KD-Tree from feature vectors and labels
    KDTree(const std::vector<std::vector<double>>& features, const std::vector<int>& labels) {
        if (features.empty() || labels.empty() || features.size() != labels.size())
            throw std::invalid_argument("Invalid training data for KDTree.");

        dims = features[0].size();
        std::vector<std::pair<std::vector<double>, int>> points;
        for (size_t i = 0; i < features.size(); ++i)
            points.emplace_back(features[i], labels[i]);

        root = build(points, 0);
    }

    // Query k nearest neighbors and return their labels
    std::vector<int> query(const std::vector<double>& target, int k) const {
        std::vector<std::pair<double, int>> best;
        best.reserve(k);
        knn_search(root.get(), target, k, best);

        std::vector<int> labels;
        for (const auto& pair : best)
            labels.push_back(pair.second);

        return labels;
    }
};

#endif // KDTREE_HPP
