from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[2]


class Sm103QKVAuxContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sm103_header = (
            REPO / "cpp/neuralnet/cudabackend_sm103.h"
        ).read_text()
        cls.sm103_source = (
            REPO / "cpp/neuralnet/cudabackend_sm103.cpp"
        ).read_text()
        cls.sm120_header = (
            REPO / "cpp/neuralnet/cudabackend_sm120.h"
        ).read_text()
        cls.sm120_source = (
            REPO / "cpp/neuralnet/cudabackend_sm120.cpp"
        ).read_text()
        cls.core = (REPO / "cpp/neuralnet/cudabackend.cpp").read_text()
        begin = cls.sm120_source.index("bool Sm120Model::sm103QKVAux(")
        end = cls.sm120_source.index("bool Sm120Model::conv1x1(", begin)
        cls.aux = cls.sm120_source[begin:end]

    def test_tactic_is_default_off_and_sm103_owned(self) -> None:
        self.assertIn('std::string qkvAuxTactic = "disabled";', self.sm103_header)
        self.assertIn(
            'std::string qkvAuxTacticSm103 = "disabled";', self.sm120_header
        )
        self.assertIn('cfg.contains("cudaSm103QKVAuxTactic")', self.sm103_source)
        self.assertIn(
            '"cublaslt-id70-q-primary-kv-aux2-b29"', self.sm103_header
        )
        self.assertIn(
            "cudaSm103QKVAuxTactic requires cudaSm103Backend=true",
            self.sm103_source,
        )
        self.assertIn(
            "cudaSm103QKVAuxTactic requires "
            "cudaSm103ReusePortableTactics=true",
            self.sm103_source,
        )

    def test_startup_is_fail_closed_on_exact_id70_and_shape(self) -> None:
        for token in (
            'activeSm120Options.qkvAuxTacticSm103 =',
            'context->sm103Options.qkvAuxTactic',
            '!activeSm120Options.useProjectionGemmLt',
            '"b29-qkv-id70-tile23-stages35-cluster5"',
            'activeSm120Options.useQKVStrided',
            'activeSm120Options.useWideQKV',
            'maxBatchSize != Sm103B29BatchSize',
            'deviceProp.major != 10 || deviceProp.minor != 3',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.core + self.sm120_source)

    def test_each_outer_model_owns_two_streams_handles_and_workspaces(self) -> None:
        state_begin = self.sm120_source.index(
            "struct Sm120Model::Sm103QKVAuxState"
        )
        state_end = self.sm120_source.index(
            "bool isSm120Arch", state_begin
        )
        state = self.sm120_source[state_begin:state_end]
        self.assertIn("cudaStream_t kStream;", state)
        self.assertIn("cudaStream_t vStream;", state)
        self.assertEqual(state.count("cudaStreamCreateWithFlags("), 2)
        self.assertEqual(state.count("cudaStreamNonBlocking"), 2)
        self.assertIn("unique_ptr<LtMatmulState> kLtState;", state)
        self.assertIn("unique_ptr<LtMatmulState> vLtState;", state)
        self.assertEqual(state.count("make_unique<LtMatmulState>(qkvTactic)"), 2)
        self.assertIn("void* workspace;", self.sm120_source)
        self.assertIn("cudaMalloc(&workspace, workspaceCapacity())", self.sm120_source)

    def test_three_reusable_events_disable_timing(self) -> None:
        state_begin = self.sm120_source.index(
            "struct Sm120Model::Sm103QKVAuxState"
        )
        state_end = self.sm120_source.index("bool isSm120Arch", state_begin)
        state = self.sm120_source[state_begin:state_end]
        for name in ("readyEvent", "kDoneEvent", "vDoneEvent"):
            self.assertIn(f"cudaEvent_t {name};", state)
        self.assertEqual(state.count("cudaEventCreateWithFlags("), 3)
        self.assertEqual(state.count("cudaEventDisableTiming"), 3)

    def test_fork_join_topology_precedes_rope(self) -> None:
        ready = self.aux.index("cudaEventRecord(\n    aux.readyEvent,primaryStream")
        k_wait = self.aux.index("cudaStreamWaitEvent(\n    aux.kStream")
        v_wait = self.aux.index("cudaStreamWaitEvent(\n    aux.vStream")
        q_launch = self.aux.index("if(!matMulLt(")
        k_launch = self.aux.index('aux.kLtState.get(),aux.kStream,"K"')
        v_launch = self.aux.index('aux.vLtState.get(),aux.vStream,"V"')
        k_done = self.aux.index("cudaEventRecord(\n    aux.kDoneEvent")
        v_done = self.aux.index("cudaEventRecord(\n    aux.vDoneEvent")
        primary_k = self.aux.index("cudaStreamWaitEvent(\n    primaryStream,aux.kDoneEvent")
        primary_v = self.aux.index("cudaStreamWaitEvent(\n    primaryStream,aux.vDoneEvent")
        self.assertLess(ready, k_wait)
        self.assertLess(ready, v_wait)
        self.assertLess(k_wait, q_launch)
        self.assertLess(v_wait, q_launch)
        self.assertLess(q_launch, k_launch)
        self.assertLess(k_launch, k_done)
        self.assertLess(k_done, v_launch)
        self.assertLess(v_launch, v_done)
        self.assertLess(v_done, primary_k)
        self.assertLess(primary_k, primary_v)
        self.assertEqual(self.aux.count("cudaEventRecord("), 3)
        self.assertEqual(self.aux.count("cudaStreamWaitEvent("), 4)

        dispatch = self.core.index("cudaHandles->sm103QKVAux(")
        rope = self.core.index("// Step 3: Apply RoPE to Q and K", dispatch)
        self.assertLess(dispatch, rope)

    def test_runtime_contract_and_private_zero_workspace_plans(self) -> None:
        for token in (
            "matBatchSize != Sm103B29Rows",
            "inputChannels != 384",
            "qChannels != 384",
            "kChannels != 384",
            "vChannels != 384",
            "!usingFP16",
            "plan->workspaceBytes != 0",
            "state->workspace",
            "plan->workspaceBytes",
            "plan->explicitSelection",
            "plan->selectionTactic != Sm103B29ProjectionGemmLtId70Tactic",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.aux)

    def test_default_and_native_paths_have_no_hook(self) -> None:
        self.assertIn("Sm103Backend::Sm103QKVAuxFn sm103QKVAux;", self.core)
        self.assertIn("sm103QKVAux(NULL)", self.core)
        self.assertIn(
            'if(activeSm120Options.qkvAuxTacticSm103 != "disabled")',
            self.core,
        )
        self.assertIn(
            "cudaHandles->sm103QKVAux = &Sm120Backend::applySm103QKVAux",
            self.core,
        )
        self.assertIn(
            'if(options.qkvAuxTacticSm103 == "disabled")\n    return false;',
            self.aux,
        )

    def test_aux_resources_retire_before_model_allocations(self) -> None:
        destructor_begin = self.sm120_source.index("Sm120Model::~Sm120Model()")
        destructor_end = self.sm120_source.index(
            "void Sm120Model::setLogger", destructor_begin
        )
        destructor = self.sm120_source[destructor_begin:destructor_end]
        reset = destructor.index("sm103QKVAuxState.reset()")
        first_free = destructor.index("cudaFree(")
        self.assertLess(reset, first_free)
        self.assertIn("cudaStreamSynchronize(kStream)", self.sm120_source)
        self.assertIn("cudaStreamSynchronize(vStream)", self.sm120_source)
        handle_destructor = self.core.index("~ComputeHandle()")
        sm103_member = self.core.index(
            "std::unique_ptr<Sm103Backend::Sm103Model> sm103Model"
        )
        self.assertGreater(handle_destructor, sm103_member)
        self.assertIn(
            "sm120Model->synchronizeSm103QKVAux()",
            self.core[handle_destructor:],
        )

    def test_teardown_is_noexcept_and_fails_fatally_on_sync_error(self) -> None:
        self.assertIn(
            "void synchronizeSm103QKVAux() noexcept;", self.sm120_header
        )
        begin = self.sm120_source.index(
            "void Sm120Model::synchronizeSm103QKVAux() noexcept"
        )
        end = self.sm120_source.index("void Sm120Model::apply(", begin)
        teardown = self.sm120_source[begin:end]
        self.assertEqual(teardown.count("cudaStreamSynchronize("), 2)
        self.assertEqual(teardown.count("Global::fatalError("), 2)
        self.assertNotIn("CUDA_ERR", teardown)
        self.assertNotIn("cudaPeekAtLastError", self.aux + teardown)

    def test_submission_apis_check_their_direct_return_status(self) -> None:
        self.assertIn("if(status != CUBLAS_STATUS_SUCCESS)", self.aux)
        self.assertEqual(self.aux.count("cudaEventRecord("), 3)
        self.assertEqual(self.aux.count("cudaStreamWaitEvent("), 4)
        self.assertGreaterEqual(self.aux.count("CUDA_ERR("), 7)


if __name__ == "__main__":
    unittest.main()
