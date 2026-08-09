#include "../tests/tests.h"

#include "../core/test.h"
#include "../neuralnet/nneval.h"

#include <chrono>
#include <future>

//------------------------
#include "../core/using.h"
//------------------------

namespace {

struct DispatchResult {
  bool gotAnything;
  size_t rows;
};

static DispatchResult waitForDispatch(
  NNBatchingDispatcher* dispatcher,
  ThreadSafeQueue<NNResultBuf*>* queue,
  int maxBatchSize,
  const std::atomic<int>* currentBatchSize,
  int serverThreadIdx
) {
  vector<NNResultBuf*> rows;
  bool gotAnything = dispatcher->waitForBatch(
    *queue,rows,maxBatchSize,*currentBatchSize,serverThreadIdx
  );
  return DispatchResult{gotAnything,rows.size()};
}

static void pushRows(ThreadSafeQueue<NNResultBuf*>& queue, int numRows) {
  for(int i = 0; i < numRows; i++)
    testAssert(queue.forcePush(NULL));
}

}

void Tests::runNNBatchingDispatcherTests() {
  cout << "Running NN batching dispatcher tests" << endl;
  const std::chrono::milliseconds shortWait(25);
  const std::chrono::seconds longWait(2);

  {
    std::atomic<int> batchSize(13);
    ThreadSafeQueue<NNResultBuf*> queue;
    NNBatchingDispatcher dispatcher(false,{0,0});
    pushRows(queue,1);
    vector<NNResultBuf*> rows;
    testAssert(dispatcher.waitForBatch(queue,rows,13,batchSize,0));
    testAssert(rows.size() == 1);
  }

  {
    std::atomic<int> batchSize(13);
    ThreadSafeQueue<NNResultBuf*> queue;
    NNBatchingDispatcher dispatcher(true,{0,0});
    std::future<DispatchResult> first = std::async(
      std::launch::async,waitForDispatch,&dispatcher,&queue,13,&batchSize,0
    );
    testAssert(first.wait_for(shortWait) == std::future_status::timeout);
    pushRows(queue,1);
    dispatcher.notify();
    testAssert(first.wait_for(longWait) == std::future_status::ready);
    DispatchResult result = first.get();
    testAssert(result.gotAnything && result.rows == 1);
    dispatcher.completeBatch(0);
  }

  {
    std::atomic<int> batchSize(13);
    ThreadSafeQueue<NNResultBuf*> queue;
    NNBatchingDispatcher dispatcher(true,{0,0});
    pushRows(queue,1);
    vector<NNResultBuf*> firstRows;
    testAssert(dispatcher.waitForBatch(queue,firstRows,13,batchSize,0));

    pushRows(queue,1);
    std::future<DispatchResult> second = std::async(
      std::launch::async,waitForDispatch,&dispatcher,&queue,13,&batchSize,1
    );
    testAssert(second.wait_for(shortWait) == std::future_status::timeout);
    pushRows(queue,12);
    dispatcher.notify();
    testAssert(second.wait_for(longWait) == std::future_status::ready);
    DispatchResult result = second.get();
    testAssert(result.gotAnything && result.rows == 13);
    dispatcher.completeBatch(1);
    dispatcher.completeBatch(0);
  }

  {
    std::atomic<int> batchSize(13);
    ThreadSafeQueue<NNResultBuf*> queue;
    NNBatchingDispatcher dispatcher(true,{0,0});
    pushRows(queue,1);
    vector<NNResultBuf*> firstRows;
    testAssert(dispatcher.waitForBatch(queue,firstRows,13,batchSize,0));
    pushRows(queue,1);
    std::future<DispatchResult> second = std::async(
      std::launch::async,waitForDispatch,&dispatcher,&queue,13,&batchSize,1
    );
    testAssert(second.wait_for(shortWait) == std::future_status::timeout);
    dispatcher.completeBatch(0);
    testAssert(second.wait_for(longWait) == std::future_status::ready);
    DispatchResult result = second.get();
    testAssert(result.gotAnything && result.rows == 1);
    dispatcher.completeBatch(1);
  }

  {
    std::atomic<int> batchSize(13);
    ThreadSafeQueue<NNResultBuf*> queue;
    NNBatchingDispatcher dispatcher(true,{0,1});
    pushRows(queue,1);
    vector<NNResultBuf*> firstRows;
    testAssert(dispatcher.waitForBatch(queue,firstRows,13,batchSize,0));
    pushRows(queue,1);
    vector<NNResultBuf*> secondRows;
    testAssert(dispatcher.waitForBatch(queue,secondRows,13,batchSize,1));
    testAssert(secondRows.size() == 1);
    dispatcher.completeBatch(1);
    dispatcher.completeBatch(0);
  }

  {
    std::atomic<int> batchSize(13);
    ThreadSafeQueue<NNResultBuf*> queue;
    NNBatchingDispatcher dispatcher(true,{0,0});
    dispatcher.resetGpuIdxByServerThread({0,1});
    pushRows(queue,1);
    vector<NNResultBuf*> firstRows;
    testAssert(dispatcher.waitForBatch(queue,firstRows,13,batchSize,0));
    pushRows(queue,1);
    vector<NNResultBuf*> secondRows;
    testAssert(dispatcher.waitForBatch(queue,secondRows,13,batchSize,1));
    testAssert(secondRows.size() == 1);
    dispatcher.completeBatch(1);
    dispatcher.completeBatch(0);
  }

  {
    std::atomic<int> batchSize(13);
    ThreadSafeQueue<NNResultBuf*> queue;
    NNBatchingDispatcher dispatcher(true,{0,0});
    pushRows(queue,1);
    vector<NNResultBuf*> firstRows;
    testAssert(dispatcher.waitForBatch(queue,firstRows,13,batchSize,0));
    pushRows(queue,4);
    std::future<DispatchResult> second = std::async(
      std::launch::async,waitForDispatch,&dispatcher,&queue,13,&batchSize,1
    );
    testAssert(second.wait_for(shortWait) == std::future_status::timeout);
    batchSize.store(4,std::memory_order_release);
    dispatcher.notify();
    testAssert(second.wait_for(longWait) == std::future_status::ready);
    DispatchResult result = second.get();
    testAssert(result.gotAnything && result.rows == 4);
    dispatcher.completeBatch(1);
    dispatcher.completeBatch(0);
  }

  {
    std::atomic<int> batchSize(13);
    ThreadSafeQueue<NNResultBuf*> queue;
    NNBatchingDispatcher dispatcher(true,{0,0});
    pushRows(queue,1);
    vector<NNResultBuf*> firstRows;
    testAssert(dispatcher.waitForBatch(queue,firstRows,13,batchSize,0));
    pushRows(queue,1);
    queue.setReadOnly();
    std::future<DispatchResult> second = std::async(
      std::launch::async,waitForDispatch,&dispatcher,&queue,13,&batchSize,1
    );
    testAssert(second.wait_for(shortWait) == std::future_status::timeout);
    dispatcher.completeBatch(0);
    testAssert(second.wait_for(longWait) == std::future_status::ready);
    DispatchResult result = second.get();
    testAssert(result.gotAnything && result.rows == 1);
    dispatcher.completeBatch(1);
    vector<NNResultBuf*> noRows;
    testAssert(!dispatcher.waitForBatch(queue,noRows,13,batchSize,0));
  }
}
