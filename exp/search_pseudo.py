async def playout(node):

    if node is root:
        if need_pause:
            await pause_gate
        timer.start()
    
    # 如果正常扩展了子节点，那就正常递归下去
    if child := expand_child(node) is not None:
        await playout(child)
        update(node, child)
        return
    
    # 已经到叶子，根据是否需要 GPU 进行分支。
    if need_nn_eval:
        h = getOutput()
        preprocess(node, h.inputs_mem_addr)
        h.input_mem_ready.set()
        # 先给一个后继任务在本线程排上队
        # 或许可以用 stdexec::task 的 sticky 语义？
        launch_task(this_thrad, playout)
        timer.stop() # 排除掉 GPU 时间
        await h.output_mem_ready.get()
        timer.start() # 排除掉 GPU 时间
        postprocess(h.outputs_mem_addr, node)
        update(node, child)
    else:
        # 不需要 GPU，那就没有 coro 调度。正常回归。
        update(node, child)

    # 回溯到根了。
    if node is root:
        timer.stop()
        update_stat(timer, thread_id) # 按之前的讨论，每个线程有一套自己的 p 和 q
        if need_nn_eval: 
            # output_mem_consumed 里面按照 pause_gate.md 的设计更新 pause_gate。
            # 为了保证 pause_gate 重新开放的时候，CPU 侧已经更新完毕，这里稍稍推迟了 output_mem_consumed
            # 置位的时间。
            h.output_mem_consumed.set() 
        else:
            # 没有 nn eval，仍然在本线程 launch 下一次 task
            launch_task(this_thread, playout)

def main():
    for thread in threads
        launch_task(thread, playout)
