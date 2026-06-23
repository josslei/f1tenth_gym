#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/simd.hpp"
#include "payload/edge_stats.hpp"
#include "payload/muzero_payload.hpp"
#include "search/mcts_search.hpp"
#include "search/muzero_search.hpp"
#include "tree/batched_mcts_tree.hpp"
#include "tree/batched_muzero_tree.hpp"
#include "tree/batched_tree_base.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tree_search_native, m) {
  m.doc() = "Native tree search bindings";

  py::class_<planner::tree_search::MuZeroSearch>(m, "MuZeroSearch")
      .def(py::init([](torch::jit::Module model, int32_t num_iters,
                       float temperature, float c_puct, int32_t batch_size,
                       int32_t action_count, int32_t hidden_size,
                       int32_t max_nodes, py::object device_obj,
                       bool print_metrics) {
             auto torch_mod = py::module_::import("torch");
             torch::Device device = [&]() {
               if (!device_obj.is_none()) {
                 return torch::Device(py::str(device_obj).cast<std::string>());
               }

               if (torch_mod.attr("cuda").attr("is_available")().cast<bool>()) {
                 return torch::Device(torch::kCUDA);
               }

               return torch::Device(torch::kCPU);
             }();
             TORCH_CHECK(
                 device.type() == torch::kCPU || device.type() == torch::kCUDA,
                 "MuZero native backend supports only CPU or CUDA, got ",
                 device.str());

             if (max_nodes <= 0) {
               max_nodes = num_iters + 1;
             }

             return planner::tree_search::MuZeroSearch(
                 std::move(model), num_iters, temperature, c_puct,
                 planner::tree_search::BatchedTreeShape(
                     batch_size, max_nodes, action_count, hidden_size),
                 device, print_metrics);
           }),
           py::arg("model"), py::arg("num_iters"), py::arg("temperature"),
           py::arg("c_puct"), py::arg("batch_size"), py::arg("action_count"),
           py::arg("hidden_size"), py::arg("max_nodes") = 0,
           py::arg("device") = py::none(), py::arg("print_metrics") = false)
      .def("search_batch", &planner::tree_search::MuZeroSearch::search_batch,
           py::arg("obs_batch"))
      .def("search_one", &planner::tree_search::MuZeroSearch::search_one,
           py::arg("obs"))
      .def("get_metrics", &planner::tree_search::MuZeroSearch::get_metrics);
}
