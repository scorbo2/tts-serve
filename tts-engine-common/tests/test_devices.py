"""Tests for the shared device grammar in ``tts_engine_common.devices``.

The grammar half of ``<PREFIX>_DEVICE`` handling lives here so no server
re-pastes its own regex (issue #73).  These tests cover the grammar and the
message-building; the *probing* half (does this box have a GPU) stays in the
servers, so it is fed in here as plain arguments.
"""

from __future__ import annotations

import pytest

from tts_engine_common.devices import (
    DEFAULT_DEVICE_FAMILIES,
    DEVICE_FAMILIES,
    INDEXABLE_FAMILIES,
    DeviceSpec,
    cuda_ordinal_error,
    parse_device,
    require_cuda_available,
)

ENV = "SOME_TTS_DEVICE"


def _parse(value: str, **kwargs) -> DeviceSpec:
    return parse_device(value, env_var=ENV, **kwargs)


class TestParseDevice:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("auto", DeviceSpec("auto", None, "auto")),
            ("cpu", DeviceSpec("cpu", None, "cpu")),
            ("mps", DeviceSpec("mps", None, "mps")),
            ("cuda", DeviceSpec("cuda", None, "cuda")),
            ("cuda:0", DeviceSpec("cuda", 0, "cuda:0")),
            ("cuda:11", DeviceSpec("cuda", 11, "cuda:11")),
        ],
    )
    def test_parse_device_wellFormedValues_describedExactly(
        self, value: str, expected: DeviceSpec
    ) -> None:
        # GIVEN a value in the default grammar,
        # WHEN parsed,
        # THEN family/ordinal/raw-text all come back as written:
        assert _parse(value) == expected

    def test_parse_device_surroundingWhitespace_isStrippedButKeptInRequested(self) -> None:
        # GIVEN an env var with copy-paste whitespace:
        # WHEN parsed,
        # THEN the grammar still matches, and the quoted text is the stripped
        # value an error message would echo back:
        spec = _parse("  cuda:1  ")
        assert (spec.family, spec.index) == ("cuda", 1)
        assert spec.requested == "cuda:1"

    def test_parse_device_defaultFamilies_advertisesTheChoiceGrammar(self) -> None:
        # GIVEN the default declaration (the grammar of the engines that offer
        # a choice -- xpu is opt-in because only one engine mentions it),
        # THEN 'auto' is a member and 'xpu' is not:
        assert DEFAULT_DEVICE_FAMILIES == ("auto", "cpu", "mps", "cuda")
        assert "xpu" not in DEFAULT_DEVICE_FAMILIES
        assert set(DEFAULT_DEVICE_FAMILIES) <= set(DEVICE_FAMILIES)

    def test_parse_device_indexableFamilies_areTheOnesTorchIndexes(self) -> None:
        # GIVEN the families an ordinal is legal for,
        # THEN CPU and MPS are absent: torch has no 'cpu:0'/'mps:0', so those
        # must be grammar errors here rather than a torch error mid-load:
        assert INDEXABLE_FAMILIES == ("cuda", "xpu")

    @pytest.mark.parametrize(
        "value",
        [
            "cuda-ish",  # the startswith("cuda") hole: begins with cuda, is not one
            "cuda:",  # an empty ordinal
            "cuda:-1",  # torch has no negative ordinal
            "cuda:1:2",
            "gpu",
            "tpu",
            "",
            "   ",
            "CPU",  # every engine behind these servers speaks torch's lowercase
            "cuda:one",
            "cpu:0",  # CPU takes no ordinal
            "mps:0",  # MPS is single-device
        ],
    )
    def test_parse_device_malformedOrUndeclaredValues_rejectedNamingTheEnvVar(
        self, value: str
    ) -> None:
        # GIVEN anything outside the declared grammar:
        # WHEN parsed,
        # THEN the ValueError names the env var, spells out the accepted
        # forms, and quotes the offending value:
        with pytest.raises(ValueError) as excinfo:
            _parse(value)
        message = str(excinfo.value)
        assert ENV in message
        assert "must be" in message
        assert repr(value) in message

    def test_parse_device_xpu_acceptedOnlyWhenTheEngineDeclaresIt(self) -> None:
        # GIVEN xpu, a family only IndexTTS's engine mentions:
        # WHEN the default declaration parses it, it is refused,
        with pytest.raises(ValueError, match=ENV):
            _parse("xpu:2")

        # WHEN a server declares it, it (and its ordinal) parse:
        assert _parse("xpu:2", families=DEVICE_FAMILIES) == DeviceSpec("xpu", 2, "xpu:2")

    @pytest.mark.parametrize("value", ["mps", "cpu", "auto"])
    def test_parse_device_undeclaredFamily_rejected(self, value: str) -> None:
        # GIVEN a CUDA-only engine's declaration,
        # WHEN any other family is parsed against it,
        # THEN it is rejected even though it is a real device family -- this is
        # the "nothing silently gains a device its engine cannot run" half:
        with pytest.raises(ValueError, match=ENV):
            _parse(value, families=("cuda",))

    def test_parse_device_cudaOnlyDeclaration_grammarListsOnlyCudaForms(self) -> None:
        # GIVEN a CUDA-only engine,
        # THEN its error message advertises exactly the two spellings it serves:
        with pytest.raises(ValueError) as excinfo:
            _parse("cpu", families=("cuda",))
        message = str(excinfo.value)
        assert "'cuda', or 'cuda:<index>'" in message
        assert "'mps'" not in message

    def test_parse_device_indexAllowed_cudaOrdinalAccepted(self) -> None:
        # GIVEN the default (index allowed),
        # THEN a second GPU is addressable:
        assert _parse("cuda:1").index == 1

    def test_parse_device_indexDisallowed_cudaOrdinalRejectedWithEngineReason(self) -> None:
        # GIVEN an engine whose device handling pins GPU 0 (LuxTTS calls
        # torch.device(device, 0), so an ordinal would be silently ignored),
        # WHEN an indexed value is parsed,
        # THEN the rejection names the env var *and* the engine's reason, so
        # the operator does not conclude the server is merely broken:
        with pytest.raises(ValueError) as excinfo:
            _parse("cuda:1", families=("cpu", "mps", "cuda"), allow_index=False, note="the engine pins GPU 0")
        message = str(excinfo.value)
        assert f"{ENV} must be" in message
        assert "'cuda:<index>'" not in message
        assert "the engine pins GPU 0" in message

    @pytest.mark.parametrize("families", [(), ("cude",), ("cuda", "gpus")])
    def test_parse_device_brokenServerDeclaration_failsLoudlyAtParseTime(
        self, families: tuple[str, ...]
    ) -> None:
        # GIVEN a server that mis-declares its own capability (a typo'd family
        # would silently widen or narrow what the server accepts),
        # WHEN parsing anything at all,
        # THEN the declaration itself is the error:
        with pytest.raises(ValueError, match=f"{ENV} is declared with unknown device families"):
            _parse("cuda", families=families)

    def test_parse_device_nonStringInput_rejectedRatherThanCrashing(self) -> None:
        # GIVEN a leaked non-string (an env var read through a JSON layer):
        # WHEN parsed,
        # THEN it is a ValueError, not an AttributeError on .strip():
        with pytest.raises(ValueError, match=ENV):
            parse_device(None, env_var=ENV)  # type: ignore[arg-type]


class TestCudaOrdinalError:
    def test_cudaOrdinalError_cudaRequestedOnBoxWithoutCuda_namesTheRequest(self) -> None:
        # GIVEN an explicit CUDA request on a box with no CUDA,
        # THEN the message says which env var asked for it and that CUDA is
        # unavailable -- torch would otherwise fail later, inside the load:
        message = cuda_ordinal_error(
            _parse("cuda"), available=False, count=0, env_var=ENV
        )
        assert message is not None
        assert f"{ENV}=cuda requests CUDA, but CUDA is not available" in message

    def test_cudaOrdinalError_indexedCudaOnBoxWithoutCuda_reportsAvailabilityFirst(
        self,
    ) -> None:
        # GIVEN an ordinal on a box with no CUDA at all,
        # THEN the complaint is the availability one (counting GPUs is beside
        # the point) rather than a confusing "0 devices" message:
        message = cuda_ordinal_error(_parse("cuda:3"), available=False, count=0, env_var=ENV)
        assert "CUDA is not available" in message
        assert "asks for GPU" not in message

    def test_cudaOrdinalError_ordinalPastTheLastGpu_namesBothNumbers(self) -> None:
        # GIVEN a one-GPU box and a request for GPU 9 (torch.device('cuda:9')
        # constructs happily and is_available() only says *some* GPU exists):
        # WHEN checked,
        # THEN the message names the requested ordinal and what the box has:
        message = cuda_ordinal_error(_parse("cuda:9"), available=True, count=1, env_var=ENV)
        assert message is not None
        assert "asks for GPU 9, but this box exposes 1 CUDA device(s)" in message

    @pytest.mark.parametrize("value", ["cuda", "cuda:0", "cuda:1", "cuda:01"])
    def test_cudaOrdinalError_ordinalWithinTheBox_returnsNone(self, value: str) -> None:
        # GIVEN a two-GPU box,
        # THEN every in-range spelling (including an un-indexed request) passes:
        assert cuda_ordinal_error(_parse(value), available=True, count=2, env_var=ENV) is None

    @pytest.mark.parametrize("value", ["auto", "cpu", "mps", "xpu", "xpu:7"])
    def test_cudaOrdinalError_nonCudaFamilies_returnsNone(self, value: str) -> None:
        # GIVEN a request that does not name CUDA -- including 'auto', which is
        # a request to *choose*, so a missing GPU is an answer rather than a
        # failure,
        # THEN CUDA is none of this function's business:
        spec = parse_device(value, env_var=ENV, families=DEVICE_FAMILIES)
        assert cuda_ordinal_error(spec, available=False, count=0, env_var=ENV) is None

    def test_cudaOrdinalError_messageNamesTheCallerEnvVarNotAGenericOne(self) -> None:
        # GIVEN two servers sharing this helper,
        # THEN each message quotes its own variable, never a shared label:
        first = cuda_ordinal_error(
            DeviceSpec("cuda", 5, "cuda:5"), available=True, count=1, env_var="ALPHA_DEVICE"
        )
        second = cuda_ordinal_error(
            DeviceSpec("cuda", 5, "cuda:5"), available=True, count=1, env_var="BETA_DEVICE"
        )
        assert "ALPHA_DEVICE=cuda:5" in first
        assert "BETA_DEVICE=cuda:5" in second
        assert "ALPHA" not in second


class TestRequireCudaAvailable:
    def test_require_cuda_available_inRangeOrdinal_returnsNone(self) -> None:
        # GIVEN a two-GPU box and an in-range request,
        # WHEN the guard runs, THEN nothing is raised:
        require_cuda_available(
            _parse("cuda:1"),
            is_available=lambda: True,
            device_count=lambda: 2,
            env_var=ENV,
        )

    def test_require_cuda_available_outOfRangeOrdinal_raisesRuntimeError(self) -> None:
        # GIVEN a request for a GPU this box does not have,
        # WHEN the guard runs,
        # THEN it raises RuntimeError (a machine problem, not a parse error)
        # with the shared wording:
        with pytest.raises(RuntimeError) as excinfo:
            require_cuda_available(
                _parse("cuda:9"),
                is_available=lambda: True,
                device_count=lambda: 1,
                env_var=ENV,
            )
        assert f"{ENV}=cuda:9 asks for GPU 9, but this box exposes 1 CUDA device" in str(
            excinfo.value
        )

    def test_require_cuda_available_nonCudaFamily_neverProbesTheBox(self) -> None:
        # GIVEN a 'cpu' request on a build whose torch has no CUDA compiled in
        # (where torch.cuda.device_count() would raise outright),
        # WHEN the guard runs,
        # THEN it never asks the box anything -- there is nothing to check:
        def explode() -> int:
            raise AssertionError("a CPU request must not probe CUDA at all")

        require_cuda_available(
            _parse("cpu"),
            is_available=explode,
            device_count=explode,
            env_var=ENV,
        )

    def test_require_cuda_available_cudaOnBoxWithoutIt_skipsTheCountProbe(
        self,
    ) -> None:
        # GIVEN a CUDA request on a box with no CUDA,
        # THEN the count is never asked for: torch.cuda.device_count() itself
        # raises on a torch build compiled without CUDA, and "how many GPUs?"
        # is a meaningless follow-up to "there are none".
        def explode() -> int:
            raise AssertionError("device_count() must not run when CUDA is absent")

        with pytest.raises(RuntimeError, match="CUDA is not available"):
            require_cuda_available(
                _parse("cuda"),
                is_available=lambda: False,
                device_count=explode,
                env_var=ENV,
            )
