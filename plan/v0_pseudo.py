# global variables

# 本轮的目标 nn 请求数量
search_nn_target_num = 0 
# 本轮的当前 nn 请求数量。当 >= 目标数量时暂停生成。
search_nn_current_num = 0
# 是否需要暂停搜索侧
search_coro_pause = False
# 暂停->继续的通知
search_coro_notice: Awaitable

class Slot:
    gpu_id: int
    stream_id: int
class InferHandle:
    cur_row: int = 0
    ready: vector<bool> # packed 
    seal: bool = False
    slot: Slot
    host_mem: bytes
    complete: Awaitable
    lock: mutex
class Request:
    handle: InferHandle
    row: int

# infer handle 的环形队列    
cur_hid = 0
cur_hid_lock: mutex
infer_handles: Array[InferHandle]

def main():
    # infer_handles 保证能容纳三倍 GPU 上最高并行量的位置数量。
    # search_nn_target_num 最多只会比 search_nn_current_num 多一倍任务总量（否则就越过了一个预测周期），所以三倍是安全的。
    infer_handles.resize(3 * sum(gpu.cuda_streams * gpu.max_batch))
    infer_thread = create_spin_wait_single_thread_stdexec_scheduler()
    for h in infer_handles:
        h.init_pinned_memory()
    for g in gpus:
        init engines, streams and cuda graphs
        
    for search_thread in range(numSearchThreads):
        new_thread.playout(root)
            
async def playout(node)
    if node is root and search_coro_pause:
        await search_coro_notice
    expand_child()
    if can_expand_child:
        await playout(child) # 遵循目前的递归写法。如果有需要可以改成非递归。
    elif need_nn_eval:
        search_nn_current_num += 1
        if search_nn_current_num >= search_nn_target_num:
            search_coro_pause = True
        this_thread.add_task(playout(root))
        await playout_gpu(data) # 需要进行一次 nn eval。为保证调度稳定，需保证返回时仍在相同线程。
        
    update(node)
    
    if node is not root:
        return
    
    # 递归回了根部，一次 playout 结束。
    update_search_coro_stats()
    if not need_nn_eval: # 保证 task 数量稳定。每次 playout 只会 launch 一次后继 playout。
        this_thread.add_task(playout(root))
    
async def playout_gpu(data)
    # 找到第一个有空位的 infer_handle，并在同一个临界区内完成 row 预留。
    cur_hid_lock.lock()
    while True:
        cur_handle = infer_handles[cur_hid]
        with cur_handle.lock:
            if cur_handle.seal:
                cur_hid = next(cur_hid)
                continue
            cur_hid_lock.unlock()
            if cur_handle.cur_row == 0: 
                cur_handle.slot = get_recent_slot(gpu_timeline)
                cur_handle.ready.resize(gpus[cur_handle.slot.gpu_id].max_batch, False)
                cur_handle.seal = False
            cur_row = cur_handle.cur_row
            cur_handle.cur_row += 1
            if cur_handle.cur_row == gpus[cur_handle.slot.gpu_id].max_batch:
                reserve_gpu_timeline()
                cur_handle.seal = True
            break
    if cur_row == 0:
        # 第一个用这个 infer_handle 的 playout，启动一下这个 infer_handle 的 infer 任务。
        infer_thread.add_task(infer_coro(cur_handle))
        
    # 到这里拿到了预处理的目标地址。
    pre_process(data, cur_handle.host_mem.get(cur_row))
    # 完成预处理了，可以通知给 infer_coro 了。
    cur_handle.ready[cur_row] = True
    # 等待 gpu 推理
    await cur_handle.complete
    # 做后处理
    post_process(cur_handle.host_mem.get(cur_row), data)
    # batch 完成后，cur_row 转为 gc counter。最后一个离开的协程负责复位 handle。
    with cur_handle.lock:
        cur_handle.cur_row -= 1
        if cur_handle.cur_row == 0:
            cur_handle.ready.fill(False)
            cur_handle.seal = False
            cur_handle.complete.reset()
    
    return data


async def infer_coro(handle):
    
    async def event_complete(event):
        if event.is_finished:
            return
        await event_complete(event) # bump back task to infinite spining loop
            
    async def ready(handle, rows_range):
        for row in rows_range
            if not handle.ready[row]:
                break
        else:
            return
        await ready(handle, rows_range) # bump back task to infinite spining loop

    def h2d(handle, rows_range, gpu, stream):
        cuda_set_device(gpu.id)
        device_mem = gpu.device_mem[stream.id]
        event = cuda_memcpy_async(handle.host_mem, rows_range, device_mem, stream) # 具体拷贝流程就不展开了。总之连续的 rows 可以压缩成单次 memcpy。
        return event
        
    async def launch(handle, batch, gpu, stream):
        cuda_set_device(gpu.id)
        graph = gpu.graphs[stream.id][batch]
        event = cuda_graph_launch_async(graph, stream)
        await event_complete(event)
        
    async def d2h(handle, batch, gpu, stream):
        cuda_set_device(gpu.id)
        device_mem = gpu.device_mem[stream.id]
        event = cuda_memcpy_async(device_mem, range(batch), handle.host_mem, stream)
        update_gpu_estimate()
        reconcile_gpu_timeline()
        update_search_nn_target_num()
        await event_complete(event)
    
    gpu = gpus[handle.slot.gpu_id]
    stream = gpu.streams[handle.slot.stream_id]
    max_batch = gpu.max_batch
    # 等待预处理完成
    row_low = 0
    
    while True:
        await sleep(0)
        with handle.lock:
            row = handle.cur_row
            if gpu.idle: # 有空闲 gpu，需要立即启动。给当前 handle 封顶，让后续的请求去下一个 handle。
                if not handle.seal:
                    reserve_gpu_timeline() # seal 当前 batch，用当前 estimate 把这次 workload 预先登记进 timeline。
                handle.seal = True
            should_break = handle.seal
        if row > row_low:
            await ready(handle, range(row_low, row))
            h2d_event = h2d(handle, range(row_low, row))
            row_low = row
        if should_break:
            break
    await event_complete(h2d_event) # 只要最后一次 event 完成，就说明前面的都完成了。
    await launch(handle, row, gpu, stream)
    await d2h(handle, row, gpu, stream)

    handle.complete.notify_all()
    
