#include <torch/script.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <string>

#include "common/simd.hpp"
#include "payload/edge_stats.hpp"
#include "payload/muzero_payload.hpp"
#include "search/mcts_search.hpp"
#include "search/muzero_search.hpp"
#include "tree/batched_mcts_tree.hpp"
#include "tree/batched_muzero_tree.hpp"
#include "tree/batched_tree_base.hpp"

namespace py = pybind11;

namespace {

torch::Tensor array_to_tensor(
    py::array_t<float, py::array::c_style | py::array::forcecast> array) {
  auto arr = array.unchecked<2>();
  return torch::from_blob(const_cast<float *>(arr.data(0, 0)),
                          {arr.shape(0), arr.shape(1)},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

py::array_t<float> tensor_to_array(const torch::Tensor &tensor) {
  if (!tensor.defined()) {
    return py::array_t<float>(0);
  }
  auto cpu = tensor.detach().to(torch::kCPU).contiguous();
  if (cpu.dim() == 1) {
    py::array_t<float> out(static_cast<py::ssize_t>(cpu.size(0)));
    std::memcpy(out.mutable_data(), cpu.data_ptr<float>(),
                static_cast<std::size_t>(cpu.numel()) * sizeof(float));
    return out;
  }
  py::array_t<float> out({static_cast<py::ssize_t>(cpu.size(0)),
                          static_cast<py::ssize_t>(cpu.size(1))});
  std::memcpy(out.mutable_data(), cpu.data_ptr<float>(),
              static_cast<std::size_t>(cpu.numel()) * sizeof(float));
  return out;
}

} // namespace

PYBIND11_MODULE(tree_search_native, m) {
  m.doc() = "Native tree search bindings";

  py::class_<planner::tree_search::MuZeroSearch>(m, "MuZeroSearch")
      .def(py::init([](const std::string &model_path, int32_t num_iters,
                       float temperature, float c_puct, int32_t batch_size,
                       int32_t action_count, int32_t hidden_size,
                       int32_t max_nodes, const std::string &device_name,
                       bool print_metrics) {
             auto torch_mod = py::module_::import("torch");
             torch::Device device = [&]() {
               if (!device_name.empty()) {
                 return torch::Device(device_name);
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

             auto model = torch::jit::load(model_path);
             model.to(device);
             model.eval();
             return planner::tree_search::MuZeroSearch(
                 std::move(model), num_iters, temperature, c_puct,
                 planner::tree_search::BatchedTreeShape(
                     batch_size, max_nodes, action_count, hidden_size),
                 device, print_metrics);
           }),
           py::arg("model_path"), py::arg("num_iters"), py::arg("temperature"),
           py::arg("c_puct"), py::arg("batch_size"), py::arg("action_count"),
           py::arg("hidden_size"), py::arg("max_nodes") = 0,
           py::arg("device") = "", py::arg("print_metrics") = false)
      .def(
          "search_batch",
          [](planner::tree_search::MuZeroSearch &self,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 obs_batch) {
            return tensor_to_array(
                self.search_batch(array_to_tensor(obs_batch)));
          },
          py::arg("obs_batch"))
      .def(
          "search_one",
          [](planner::tree_search::MuZeroSearch &self,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 obs) {
            auto arr = obs.unchecked<1>();
            auto tensor = torch::from_blob(
                              const_cast<float *>(arr.data(0)), {arr.shape(0)},
                              torch::TensorOptions().dtype(torch::kFloat32))
                              .clone();
            return tensor_to_array(self.search_one(tensor));
          },
          py::arg("obs"))
      .def("get_metrics", &planner::tree_search::MuZeroSearch::get_metrics);
}
