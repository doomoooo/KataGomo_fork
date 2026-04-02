#include "schedlab/infer_backend.hpp"

#include <cstring>

namespace schedlab {
  InferBackend::~InferBackend() {
    release_host_slots();
  }

  void InferBackend::allocate_host_slots(std::uint32_t host_slot_count) {
    const std::size_t input_slab_bytes = batch_layout.input_slab_bytes;
    const std::size_t output_slab_bytes = batch_layout.output_slab_bytes;
    const std::size_t total_bytes = input_slab_bytes + output_slab_bytes;
    release_host_slots();
    for(std::uint32_t slot_index = 0; slot_index < host_slot_count; ++slot_index) {
      HostSlot host_slot;
      host_slot.raw_storage = allocate_raw_host_storage(total_bytes);

      auto* base = static_cast<char*>(host_slot.raw_storage);
      host_slot.input_slab = base;
      host_slot.output_slab = base + input_slab_bytes;
      host_slot.inputs.resize(batch_layout.input_row_bytes.size());
      host_slot.outputs.resize(batch_layout.output_row_bytes.size());

      for(std::size_t tensor_index = 0; tensor_index < batch_layout.input_tensor_offsets.size(); ++tensor_index) {
        host_slot.inputs[tensor_index] =
          static_cast<char*>(host_slot.input_slab) + batch_layout.input_tensor_offsets[tensor_index];
      }
      for(std::size_t tensor_index = 0; tensor_index < batch_layout.output_tensor_offsets.size(); ++tensor_index) {
        host_slot.outputs[tensor_index] =
          static_cast<char*>(host_slot.output_slab) + batch_layout.output_tensor_offsets[tensor_index];
      }

      host_slots.push_back(std::move(host_slot));
    }
  }

  void InferBackend::release_host_slots() noexcept {
    for(HostSlot& host_slot: host_slots) {
      if(host_slot.raw_storage) {
        release_raw_host_storage(host_slot.raw_storage);
      }
      host_slot = {};
    }
    host_slots.clear();
  }
}  // namespace schedlab
