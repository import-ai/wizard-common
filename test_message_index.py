import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from wizard_common.grimoire.entity.message import Message, OpenAIMessage
from wizard_common.grimoire.retriever.weaviate_vector_db import WeaviateVectorDB
from wizard_common.worker.retry import RetryableTaskError


def test_message_batch_failure_and_chunk_roundtrip():
    async def check():
        db = object.__new__(WeaviateVectorDB)
        db.batch_size = 2
        db._embed = AsyncMock(return_value=[[0.1]])
        db.remove_message_vectors = AsyncMock()
        insert = AsyncMock(
            return_value=SimpleNamespace(has_errors=True, errors={0: "error"})
        )
        db._get_shard = AsyncMock(
            return_value=SimpleNamespace(data=SimpleNamespace(insert_many=insert))
        )
        msg = Message(
            conversation_id="c",
            message_id="m",
            message=OpenAIMessage(role="assistant", content="尾部事实"),
        )
        try:
            await db.upsert_message("ns", "user", msg)
        except RetryableTaskError:
            pass
        else:
            raise AssertionError("Partial batch failure must escape")
        props = insert.call_args.args[0][0].properties
        recovered = db._message_from_flat_doc(props)
        assert recovered.message.content == "尾部事实"
        assert (recovered.chunk_index, recovered.start_index, recovered.end_index) == (
            0,
            0,
            4,
        )
        db.remove_message_vectors.reset_mock()
        db._embed.return_value = []
        try:
            await db.upsert_message("ns", "user", msg)
        except RetryableTaskError:
            pass
        else:
            raise AssertionError("Missing embedding must fail")
        db.remove_message_vectors.assert_not_called()

    asyncio.run(check())


def test_message_embeddings_obey_batch_size_before_replacing_vectors():
    async def check():
        db = object.__new__(WeaviateVectorDB)
        db.batch_size = 2
        db._embed = AsyncMock(side_effect=[[[1.0], [2.0]], [[3.0]]])
        db.remove_message_vectors = AsyncMock()
        insert = AsyncMock(return_value=SimpleNamespace(has_errors=False))
        db._get_shard = AsyncMock(
            return_value=SimpleNamespace(data=SimpleNamespace(insert_many=insert))
        )
        message = Message(
            conversation_id="c",
            message_id="m",
            message=OpenAIMessage(role="assistant", content="one\n\ntwo\n\nthree"),
        )
        await db.upsert_message("ns", "user", message)
        assert [call.args[0] for call in db._embed.call_args_list] == [
            ["one", "two"],
            ["three"],
        ]
        assert [obj.vector for obj in insert.call_args.args[0]] == [[1.0], [2.0], [3.0]]
        db.remove_message_vectors.assert_awaited_once()
        db.remove_message_vectors.reset_mock()
        insert.reset_mock()
        db._embed.side_effect = [[[1.0], [2.0]], []]
        try:
            await db.upsert_message("ns", "user", message)
        except RetryableTaskError:
            pass
        else:
            raise AssertionError("A later batch failure must preserve old vectors")
        db.remove_message_vectors.assert_not_awaited()
        insert.assert_not_awaited()

    asyncio.run(check())
