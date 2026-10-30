import os

import pytest

from whispbook_gender_specdec.checkpoints import discover_local_checkpoints
from whispbook_gender_specdec.mlx_runner import infer_with_mlx_speculative
from whispbook_gender_specdec.speculative import choose_draft_checkpoint


pytestmark = pytest.mark.live


@pytest.mark.parametrize("family", ["smollm", "gemma", "qwen"])
def test_live_local_checkpoint_infers_gender_with_speculative_decoding(family: str) -> None:
    if os.environ.get("RUN_LIVE_MLX_GENDER") != "1":
        pytest.skip("set RUN_LIVE_MLX_GENDER=1 to load local MLX checkpoints")

    manifest = discover_local_checkpoints()
    target = manifest[family]
    if not target.available:
        pytest.skip(f"{family} checkpoint is not in the local cache: {target.notes}")

    draft = choose_draft_checkpoint(target, manifest.values())
    result = infer_with_mlx_speculative(
        target=target,
        draft=draft.spec,
        text="I will not abandon the castle now, said Maria.",
        source_hint="live_test:maria",
        max_tokens=64,
        num_draft_tokens=3,
    )

    assert result.gender in {"male", "female", "unknown"}, result.raw_response
    if family == "qwen":
        assert result.gender == "female", result.raw_response
    assert result.speculative.target_family == family
    assert result.speculative.draft_family == draft.spec.family
    assert result.speculative.total_tokens > 0
    assert result.speculative.draft_tokens > 0
