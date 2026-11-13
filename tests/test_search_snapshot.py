import importlib.util
import sys
from pathlib import Path

import pytest

from wizard_common.grimoire.entity.tools import Condition

spec = importlib.util.spec_from_file_location(
    "snapshot_fakes", Path(__file__).with_name("test_weaviate_hybrid_fusion.py")
)
fakes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fakes
spec.loader.exec_module(fakes)


@pytest.mark.asyncio
async def test_snapshot_fuses_all_branches_once_and_excludes_lookahead():
    db, col = fakes.make_db(
        vector_objects=[fakes.obj(i, distance=0.1 + i * 0.001) for i in range(1, 5)],
        word_objects=[fakes.obj(i, score=10 - i * 0.01) for i in range(5, 9)],
    )
    result = await db._retrieve_window(
        "space", "code", Condition(namespace_id="space"), limit=3, snapshot=True
    )
    assert {c.uuid for c in result.candidates} == {
        str(fakes.uuid.UUID(int=i)) for i in [1, 2, 3, 5, 6, 7]
    }
    assert result.has_more
    assert col.query.near_vector.call_args.kwargs["limit"] == 4
    db._embed.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_rejects_partial_branch_failure_before_publishing():
    db, col = fakes.make_db(word_objects=[fakes.obj(1, score=5)])
    col.query.near_vector.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError):
        await db._retrieve_window(
            "space", "code", Condition(namespace_id="space"), limit=3, snapshot=True
        )


@pytest.mark.asyncio
async def test_lookahead_score_does_not_look_like_a_failed_threshold():
    target = fakes.obj(1, distance=0.4)
    db, _ = fakes.make_db(
        vector_objects=[target],
        word_objects=[
            fakes.obj(2, score=5),
            fakes.obj(3, score=4),
            fakes.obj(1, score=3),
        ],
    )
    result = await db._retrieve_window(
        "space", "code", Condition(namespace_id="space"), limit=2, snapshot=True
    )
    candidate = next(c for c in result.candidates if c.uuid == str(target.uuid))
    assert candidate.passed_legs == ["vector"]
    assert candidate.word_bm25_score is None
    assert "word" not in candidate.returned_legs
