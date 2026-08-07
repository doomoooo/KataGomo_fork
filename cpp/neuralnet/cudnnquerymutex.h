#ifndef KATAGO_CUDNN_QUERY_MUTEX_H
#define KATAGO_CUDNN_QUERY_MUTEX_H

#include <mutex>

namespace CudaBackendInternal {

// Some cuDNN releases are not safe when independent NN-server initialization
// threads query legacy convolution algorithms concurrently on one process.
// The query runs only while constructing a model, never during inference.
inline std::mutex& cudnnConvolutionAlgorithmQueryMutex() {
  static std::mutex mutex;
  return mutex;
}

} // namespace CudaBackendInternal

#endif
