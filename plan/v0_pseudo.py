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
    slot: Slot
    host_mem: bytes
    complete: Awaitable
    lock: mutex
class Request:
    handle: InferHandle
    row: int

# infer handle 的环形队列    
cur_hid = 0
infer_handles: Array[InferHandle]

# 动态更新的 recent_slot，更新时间点在任意 infer launch 和 finish
recent_slot: Slot = [0, 0] # 需要是一个合法初值。这里只是示意。gpu 0 不一定在。
# 保证 recent_slot 和实际 GPU 状态原子更新的锁。
recent_slot_lock: mutex

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
    
    expand_child()
    if can_expand_child:
        playout(child) # 遵循目前的递归写法。如果有需要可以改成非递归。
    elif need_nn_eval:
        search_nn_target_num += 1
        if search_nn_target_num >= search_nn_current_num:
            search_coro_pause = True
        this_thread.add_task(playout(root))
        await gpu(data) # 需要进行一次 nn eval。为保证调度稳定，需保证返回时仍在相同线程。
        
    update(node)
    
    if node is not root:
        return
    
    # 递归回了根部，一次 playout 结束。
    update_search_coro_stats()
    if search_coro_pause:
        await search_coro_notice
    if not need_nn_eval: # 保证 task 数量稳定。每次 playout 只会 launch 一次后继 playout。
        this_thread.add_task(playout(root))
    
async def playout_gpu(data)
    # 找到第一个有空位的 infer_handle
    while True:
        cur_handle = infer_handles[cur_hid]
        with cur_handle.lock:
            if cur_handle.cur_row == max_batch: # 这个 infer batch 已经封顶
                cur_hid = next(cur_hid)
        break
    # 查看当前是否为空 batch，需要选择 slot
    with cur_handle.lock:
        if cur_handle.cur_row == 0: 
            with recent_slot_lock:
                cur_handle.slot = recent_slot
        cur_handle.ready.resize(cur_handle.slot.gpu.batch_size, False)
        cur_row = cur_handle.cur_row
        cur_handle.cur_row += 1
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
        update_gpu_estimate() # 如果同一个 GPU 上有其它活动流，多启动一个流回压缩其它流的工作效率。所以需要更新。
        update_recent_slot()
        await event_complete(event)
        
    async def d2h(handle, batch, gpu, stream):
        cuda_set_device(gpu.id)
        device_mem = gpu.device_mem[stream.id]
        event = cuda_memcpy_async(device_mem, range(batch), handle.host_mem, stream)
        update_gpu_estimate()
        update_gpu_idle_state() # 这个位置可以获得所有 GPU 空闲的消息。
        update_recent_slot()
        update_search_nn_target_num()
        await event_complete(event)
        
    
    gpu = gpus[handle.slot.gpu_id]
    stream = gpu.streams[handle.slot.stream_id]
    max_batch = gpu.max_batch
    # 等待预处理完成
    row_low = 0
    
    while True:
        with handle.lock:
            row = handle.cur_row
            if gpu.idle: # 有空闲 gpu，需要立即启动。给当前 handle 封顶，让后续的请求去下一个 handle。
                is_idle = True
                handle.cur_row = max_batch
        await ready(handle, range(row_low, row))
        h2d_event = h2d(handle, range(row_low, row))
        row_low = row
        if is_idle or row == batch_size:
            break
    await event_complete(h2d_event) # 只要最后一次 event 完成，就说明前面的都完成了。
    
    await launch(handle, row, gpu, stream)
    await d2h(handle, row, gpu, stream)

    handle.complete.notify_all()
    