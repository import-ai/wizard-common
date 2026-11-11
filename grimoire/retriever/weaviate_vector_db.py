import asyncio
import re
from functools import partial
from typing import Any, List, Tuple
from uuid import NAMESPACE_URL, uuid5

import weaviate
import weaviate.classes as wvc
from common.trace_info import TraceInfo
from openai import AsyncOpenAI
from opentelemetry import propagate, trace
from pydantic import BaseModel
from weaviate.exceptions import (
    WeaviateDeleteManyError,
    WeaviateQueryError,
)

from wizard_common.grimoire.config import (
    SearchDiagnosticsConfig,
    SearchThresholdsConfig,
    VectorConfig,
)
from wizard_common.grimoire.entity.chunk import Chunk, ResourceChunkRetrieval
from wizard_common.grimoire.entity.index_record import IndexRecord, IndexRecordType
from wizard_common.grimoire.entity.message import Message, OpenAIMessage
from wizard_common.grimoire.entity.retrieval import Score
from wizard_common.grimoire.entity.tools import (
    Condition,
    PrivateSearchResourceType,
    PrivateSearchTool,
    Resource,
)
from wizard_common.grimoire.retriever.base import BaseRetriever, SearchFunction
from wizard_common.grimoire.retriever.search_diagnostics import (
    distance_metric,
    record_candidates,
)
from wizard_common.grimoire.retriever.search_retrieval import (
    RetrievalWindow,
    RetrievedCandidate,
    retrieve_window,
)
from wizard_common.worker.retry import RetryableTaskError

tracer = trace.get_tracer(__name__)
COLLECTION_NAME = "omnibox_index"
MESSAGE_CHUNK_SIZE = 2000
WORD_QUERY_PROPERTIES = [
    "chunk_title^3",
    "chunk_text",
    "message_content",
    "resource_tag_names",
]
GSE_QUERY_PROPERTIES = [
    "chunk_title_gse^3",
    "chunk_text_gse",
    "message_content_gse",
    "resource_tag_names_gse",
]


def split_message_content(content: str, chunk_size: int = MESSAGE_CHUNK_SIZE):
    chunks: list[tuple[str, int, int]] = []
    cursor = 0
    for paragraph in content.split("\n\n"):
        start = cursor
        cursor += len(paragraph) + 2
        if not paragraph:
            continue
        for offset in range(0, len(paragraph), chunk_size):
            text = paragraph[offset : offset + chunk_size]
            chunks.append((text, start + offset, start + offset + len(text)))
    return chunks


class VectorSearchPage(BaseModel):
    records: list[IndexRecord]
    has_more: bool


class WeaviateVectorDB:
    _english_letters_pattern = re.compile(r"[0-9A-Za-z]+")
    _whitespace_pattern = re.compile(r"\s+")

    def __init__(self, config: VectorConfig):
        self.config: VectorConfig = config
        self.batch_size: int = config.batch_size
        self.openai = AsyncOpenAI(
            api_key=config.embedding.api_key, base_url=config.embedding.base_url
        )
        self.client: weaviate.WeaviateAsyncClient = ...
        self._init_lock = asyncio.Lock()
        self.dimension = config.dimension
        self._distance_metric = "unknown"

    async def _ensure_client(self) -> None:
        if self.client is not ...:
            return
        async with self._init_lock:
            if self.client is not ...:
                return

            connect_kwargs: dict[str, Any] = {
                "port": self.config.weaviate.port,
                "grpc_port": self.config.weaviate.grpc_port,
            }
            if self.config.weaviate.api_key:
                connect_kwargs["auth_credentials"] = wvc.init.Auth.api_key(
                    self.config.weaviate.api_key
                )
            if self.config.weaviate.host:
                connect_kwargs["host"] = self.config.weaviate.host
            client = weaviate.use_async_with_local(**connect_kwargs)
            await client.connect()
            self.client = client

            if await client.collections.exists(COLLECTION_NAME):
                await self._ensure_collection_properties()
            else:
                await self._create_collection()

    @staticmethod
    def _required_collection_properties() -> list[wvc.config.Property]:
        return [
            wvc.config.Property(
                name="type",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=True,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="namespace_id",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=True,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="user_id",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=True,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="chunk_title_gse",
                data_type=wvc.config.DataType.TEXT,
                index_searchable=True,
                tokenization=wvc.config.Tokenization.GSE_CH,
            ),
            wvc.config.Property(
                name="chunk_title",
                data_type=wvc.config.DataType.TEXT,
                index_searchable=True,
                tokenization=wvc.config.Tokenization.WORD,
            ),
            wvc.config.Property(
                name="chunk_text_gse",
                data_type=wvc.config.DataType.TEXT,
                index_searchable=True,
                tokenization=wvc.config.Tokenization.GSE_CH,
            ),
            wvc.config.Property(
                name="chunk_text",
                data_type=wvc.config.DataType.TEXT,
                index_searchable=True,
                tokenization=wvc.config.Tokenization.WORD,
            ),
            wvc.config.Property(
                name="chunk_resource_id",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=True,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="chunk_parent_id",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=True,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="resource_tag_ids",
                data_type=wvc.config.DataType.TEXT_ARRAY,
                index_filterable=True,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="resource_tag_names_gse",
                data_type=wvc.config.DataType.TEXT_ARRAY,
                index_searchable=True,
                tokenization=wvc.config.Tokenization.GSE_CH,
            ),
            wvc.config.Property(
                name="resource_tag_names",
                data_type=wvc.config.DataType.TEXT_ARRAY,
                index_searchable=True,
                tokenization=wvc.config.Tokenization.WORD,
            ),
            wvc.config.Property(
                name="chunk_type",
                data_type=wvc.config.DataType.TEXT,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="chunk_id",
                data_type=wvc.config.DataType.TEXT,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="chunk_start_index",
                data_type=wvc.config.DataType.INT,
            ),
            wvc.config.Property(
                name="chunk_end_index",
                data_type=wvc.config.DataType.INT,
            ),
            wvc.config.Property(
                name="chunk_line_range",
                data_type=wvc.config.DataType.TEXT,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="chunk_created_at",
                data_type=wvc.config.DataType.NUMBER,
                index_filterable=True,
                index_range_filters=True,
            ),
            wvc.config.Property(
                name="chunk_updated_at",
                data_type=wvc.config.DataType.NUMBER,
                index_filterable=True,
                index_range_filters=True,
            ),
            wvc.config.Property(
                name="message_id",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=True,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="conversation_id",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=True,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="message_role",
                data_type=wvc.config.DataType.TEXT,
                tokenization=wvc.config.Tokenization.FIELD,
            ),
            wvc.config.Property(
                name="message_content_gse",
                data_type=wvc.config.DataType.TEXT,
                index_searchable=True,
                tokenization=wvc.config.Tokenization.GSE_CH,
            ),
            wvc.config.Property(
                name="message_content",
                data_type=wvc.config.DataType.TEXT,
                index_searchable=True,
                tokenization=wvc.config.Tokenization.WORD,
            ),
            wvc.config.Property(
                name="message_chunk_index",
                data_type=wvc.config.DataType.INT,
            ),
            wvc.config.Property(
                name="message_chunk_start_index",
                data_type=wvc.config.DataType.INT,
            ),
            wvc.config.Property(
                name="message_chunk_end_index",
                data_type=wvc.config.DataType.INT,
            ),
        ]

    async def _ensure_collection_properties(self) -> None:
        required_properties = self._required_collection_properties()
        collection = self.client.collections.get(COLLECTION_NAME)
        config = await collection.config.get()
        self._distance_metric = distance_metric(config)
        existing_names = {prop.name for prop in config.properties}
        for prop in required_properties:
            if prop.name not in existing_names:
                await collection.config.add_property(prop)

    async def _create_collection(self) -> None:
        required_properties = self._required_collection_properties()
        await self.client.collections.create(
            name=COLLECTION_NAME,
            vector_config=wvc.config.Configure.Vectors.self_provided(),
            multi_tenancy_config=wvc.config.Configure.multi_tenancy(
                enabled=True, auto_tenant_creation=True
            ),
            inverted_index_config=wvc.config.Configure.inverted_index(
                index_null_state=True,
                bm25_b=0.75,
                bm25_k1=1.2,
            ),
            properties=required_properties,
        )

    @classmethod
    def _strip_english_letters(cls, value: str) -> str:
        without_english = cls._english_letters_pattern.sub(" ", value)
        return cls._whitespace_pattern.sub(" ", without_english).strip()

    async def _get_shard(self, namespace_id: str):
        if not namespace_id:
            raise ValueError("namespace_id is required")
        await self._ensure_client()
        collection = self.client.collections.get(COLLECTION_NAME)
        return collection.with_tenant(namespace_id)

    async def _embed(self, input_: str | list[str]) -> list[list[float]]:
        headers = {}
        propagate.inject(headers)
        embeddings = await self.openai.embeddings.create(
            model=self.config.embedding.model, input=input_, extra_headers=headers
        )
        return [item.embedding for item in embeddings.data]

    @property
    def retrieval_signature(self) -> str:
        return (
            "split-thresholds-title3-v2:"
            + self.config.search_thresholds.model_dump_json()
        )

    @tracer.start_as_current_span("WeaviateVectorDB._hybrid_query")
    async def _retrieve_window(
        self,
        namespace_id: str,
        query: str,
        condition: Condition,
        limit: int,
        offset: int = 0,
    ) -> RetrievalWindow:
        collection = await self._get_shard(namespace_id)
        filters = condition.to_weaviate_filters()
        span = trace.get_current_span()
        span.set_attribute("weaviate.hybrid.condition", condition.model_dump_json())
        if not query:
            try:
                response = await collection.query.fetch_objects(
                    filters=filters,
                    limit=offset + limit + 1,
                    return_metadata=wvc.query.MetadataQuery.full(),
                )
            except WeaviateQueryError as error:
                if "tenant not found" in str(error).lower():
                    return RetrievalWindow()
                raise
            candidates = [
                RetrievedCandidate(
                    uuid=str(obj.uuid),
                    properties=obj.properties or {},
                    final_score=float(getattr(obj.metadata, "score", None) or 0),
                )
                for obj in response.objects
            ]
            candidates.sort(key=lambda c: c.final_score, reverse=True)
            return RetrievalWindow(
                candidates=candidates[offset : offset + limit],
                has_more=len(candidates) > offset + limit,
            )
        config = getattr(self, "config", None)
        thresholds = config.search_thresholds if config else SearchThresholdsConfig()
        window = await retrieve_window(
            collection=collection,
            embed=self._embed,
            query=query,
            gse_query=self._strip_english_letters(query),
            filters=filters,
            thresholds=thresholds,
            word_properties=WORD_QUERY_PROPERTIES,
            gse_properties=GSE_QUERY_PROPERTIES,
            offset=offset,
            limit=limit,
            span=span,
        )
        diagnostics = config.search_diagnostics if config else SearchDiagnosticsConfig()
        if diagnostics.enabled and span.is_recording():
            try:
                record_candidates(
                    span=span,
                    config=diagnostics,
                    window=window,
                    metric=getattr(self, "_distance_metric", "unknown"),
                    offset=offset,
                )
            except Exception:
                span.set_attribute("search.diagnostics.status", "error")
        return window

    async def _hybrid_query(
        self,
        namespace_id: str,
        query: str,
        condition: Condition,
        limit: int,
        offset: int = 0,
    ) -> List[Tuple[dict, float]]:
        window = await self._retrieve_window(
            namespace_id, query, condition, limit, offset
        )
        return [(c.properties, c.final_score) for c in window.candidates]

    @staticmethod
    def _chunk_from_flat_doc(doc: dict) -> Chunk:
        return Chunk(
            title=doc.get("chunk_title"),
            resource_id=doc["chunk_resource_id"],
            text=doc.get("chunk_text"),
            chunk_type=doc["chunk_type"],
            parent_id=doc["chunk_parent_id"],
            chunk_id=doc["chunk_id"],
            created_at=doc["chunk_created_at"],
            updated_at=doc["chunk_updated_at"],
            start_index=doc.get("chunk_start_index"),
            end_index=doc.get("chunk_end_index"),
            line_range=doc.get("chunk_line_range"),
            resource_tag_ids=doc.get("resource_tag_ids") or [],
            resource_tag_names=doc.get("resource_tag_names") or [],
        )

    @staticmethod
    def _message_from_flat_doc(doc: dict) -> Message:
        return Message(
            chunk_index=doc.get("message_chunk_index"),
            start_index=doc.get("message_chunk_start_index"),
            end_index=doc.get("message_chunk_end_index"),
            conversation_id=doc["conversation_id"],
            message_id=doc["message_id"],
            message=OpenAIMessage(
                role=doc.get("message_role") or "assistant",
                content=doc.get("message_content") or "",
            ),
        )

    @classmethod
    def _index_record_from_flat_doc(cls, doc: dict) -> IndexRecord:
        record_type = doc.get("type")
        if record_type == IndexRecordType.chunk.value:
            return IndexRecord(
                type=IndexRecordType.chunk,
                namespace_id=doc["namespace_id"],
                user_id=doc.get("user_id"),
                chunk=cls._chunk_from_flat_doc(doc),
            )
        if record_type == IndexRecordType.message.value:
            return IndexRecord(
                type=IndexRecordType.message,
                namespace_id=doc["namespace_id"],
                user_id=doc.get("user_id"),
                message=cls._message_from_flat_doc(doc),
            )
        return IndexRecord(**doc)

    @tracer.start_as_current_span("WeaviateVectorDB.insert_chunks")
    async def insert_chunks(self, namespace_id: str, chunk_list: List[Chunk]):
        collection = await self._get_shard(namespace_id)

        for i in range(0, len(chunk_list), self.batch_size):
            raw_batch = chunk_list[i : i + self.batch_size]
            batch: List[Chunk] = []
            prompts: list[str] = []
            for x in raw_batch:
                prompt = x.to_prompt()
                if prompt:
                    batch.append(x)
                    prompts.append(prompt)
            if not batch:
                continue

            vectors = await self._embed(prompts)
            objects = []
            for chunk, vector in zip(batch, vectors):
                properties = {
                    "type": IndexRecordType.chunk.value,
                    "namespace_id": namespace_id,
                }
                properties["chunk_title"] = chunk.title
                properties["chunk_title_gse"] = self._strip_english_letters(chunk.title)
                properties["chunk_text"] = chunk.text
                properties["chunk_text_gse"] = self._strip_english_letters(chunk.text)
                properties["chunk_resource_id"] = chunk.resource_id
                properties["chunk_parent_id"] = chunk.parent_id
                properties["resource_tag_ids"] = chunk.resource_tag_ids
                properties["resource_tag_names"] = chunk.resource_tag_names
                properties["resource_tag_names_gse"] = [
                    self._strip_english_letters(name)
                    for name in chunk.resource_tag_names
                ]
                properties["chunk_type"] = chunk.chunk_type.value
                properties["chunk_id"] = chunk.chunk_id
                properties["chunk_created_at"] = chunk.created_at
                properties["chunk_updated_at"] = chunk.updated_at
                properties["chunk_start_index"] = chunk.start_index
                properties["chunk_end_index"] = chunk.end_index
                properties["chunk_line_range"] = chunk.line_range
                objects.append(
                    wvc.data.DataObject(
                        properties=properties,
                        vector=vector,
                    )
                )
            await collection.data.insert_many(objects)

    @tracer.start_as_current_span("WeaviateVectorDB.upsert_message")
    async def upsert_message(self, namespace_id: str, user_id: str, message: Message):
        collection = await self._get_shard(namespace_id)
        message_content = message.message.content
        chunks = split_message_content(message_content)
        vectors = []
        for offset in range(0, len(chunks), self.batch_size):
            batch = chunks[offset : offset + self.batch_size]
            batch_vectors = await self._embed([chunk[0] for chunk in batch])
            if len(batch_vectors) != len(batch):
                raise RetryableTaskError(
                    "Message embedding count does not match chunk count"
                )
            vectors.extend(batch_vectors)
        await self.remove_message_vectors(namespace_id, message_id=message.message_id)
        objects = []
        for index, ((text, start, end), vector) in enumerate(zip(chunks, vectors)):
            properties = {
                "type": IndexRecordType.message.value,
                "namespace_id": namespace_id,
                "user_id": user_id,
                "message_id": message.message_id,
                "conversation_id": message.conversation_id,
                "message_role": message.message.role,
                "message_content": text,
                "message_content_gse": self._strip_english_letters(text),
                "message_chunk_index": index,
                "message_chunk_start_index": start,
                "message_chunk_end_index": end,
            }
            objects.append(
                wvc.data.DataObject(
                    properties=properties,
                    vector=vector,
                    uuid=uuid5(
                        NAMESPACE_URL, f"{namespace_id}:{message.message_id}:{index}"
                    ),
                )
            )
        if objects:
            result = await collection.data.insert_many(objects)
            if result.has_errors:
                raise RetryableTaskError(
                    f"Message vector batch failed: {len(result.errors)} objects"
                )

    async def remove_message_vectors(
        self, namespace_id: str, message_id: str | None = None
    ):
        collection = await self._get_shard(namespace_id)
        filters = wvc.query.Filter.by_property("type").equal(
            IndexRecordType.message.value
        ) & wvc.query.Filter.by_property("namespace_id").equal(namespace_id)
        if message_id:
            filters &= wvc.query.Filter.by_property("message_id").equal(message_id)
        deleted = 0
        while True:
            try:
                result = await collection.data.delete_many(where=filters)
            except WeaviateDeleteManyError as error:
                if "tenant not found" in str(error).lower():
                    return deleted
                raise
            if result.failed:
                raise RetryableTaskError(
                    f"Message vector deletion failed: {result.failed} objects"
                )
            deleted += result.successful
            remaining = await collection.query.fetch_objects(filters=filters, limit=1)
            if not remaining.objects:
                return deleted
            if not result.successful:
                raise RetryableTaskError("Message vector deletion made no progress")

    @tracer.start_as_current_span("WeaviateVectorDB.remove_conversation")
    async def remove_conversation(self, namespace_id: str, conversation_id: str):
        collection = await self._get_shard(namespace_id)
        try:
            await collection.data.delete_many(
                where=wvc.query.Filter.by_property("type").equal(
                    IndexRecordType.message.value
                )
                & wvc.query.Filter.by_property("namespace_id").equal(namespace_id)
                & wvc.query.Filter.by_property("conversation_id").equal(conversation_id)
            )
        except WeaviateDeleteManyError:
            # Tenant not found (no data yet for this namespace)
            pass

    @tracer.start_as_current_span("WeaviateVectorDB.remove_chunks")
    async def remove_chunks(self, namespace_id: str, resource_id: str):
        collection = await self._get_shard(namespace_id)
        try:
            await collection.data.delete_many(
                where=wvc.query.Filter.by_property("type").equal(
                    IndexRecordType.chunk.value
                )
                & wvc.query.Filter.by_property("namespace_id").equal(namespace_id)
                & wvc.query.Filter.by_property("chunk_resource_id").equal(resource_id)
            )
        except WeaviateDeleteManyError:
            # Tenant not found (no data yet for this namespace)
            pass

    async def search_page(
        self,
        query: str,
        namespace_id: str,
        user_id: str | None,
        record_type: IndexRecordType | None,
        offset: int,
        limit: int,
        conversation_ids: list[str] | None = None,
        exclude_conversation_ids: list[str] | None = None,
        resource_ids: list[str] | None = None,
    ) -> VectorSearchPage:
        condition = Condition(
            conversation_ids=conversation_ids,
            exclude_conversation_ids=exclude_conversation_ids,
            namespace_id=namespace_id,
            user_id=user_id,
            record_type=record_type.value if record_type else None,
            resource_ids=resource_ids,
        )
        window = await self._retrieve_window(
            namespace_id, query, condition, limit, offset
        )
        return VectorSearchPage(
            records=[
                self._index_record_from_flat_doc(c.properties)
                for c in window.candidates
            ],
            has_more=window.has_more,
        )

    @tracer.start_as_current_span("WeaviateVectorDB.search")
    async def search(
        self,
        query: str,
        namespace_id: str,
        user_id: str | None,
        record_type: IndexRecordType | None,
        offset: int,
        limit: int,
        conversation_ids: list[str] | None = None,
        exclude_conversation_ids: list[str] | None = None,
        resource_ids: list[str] | None = None,
    ) -> List[IndexRecord]:
        page = await self.search_page(
            query,
            namespace_id,
            user_id,
            record_type,
            offset,
            limit,
            conversation_ids,
            exclude_conversation_ids,
            resource_ids,
        )
        return page.records

    @tracer.start_as_current_span("WeaviateVectorDB.query_chunks")
    async def query_chunks(
        self,
        namespace_id: str,
        query: str,
        k: int,
        condition: Condition,
    ) -> List[Tuple[Chunk, float]]:
        combined_condition = condition.model_copy(
            update={"record_type": IndexRecordType.chunk.value}
        )
        hits = await self._hybrid_query(
            namespace_id=namespace_id,
            query=query,
            condition=combined_condition,
            limit=k,
        )
        output: List[Tuple[Chunk, float]] = []
        for hit, score in hits:
            chunk = self._chunk_from_flat_doc(hit)
            output.append((chunk, score))
        return output


class WeaviateVectorRetriever(BaseRetriever):
    def __init__(self, config: VectorConfig):
        self.vector_db = WeaviateVectorDB(config)

    @staticmethod
    def get_folder(resource_id: str, resources: list[Resource]) -> str | None:
        for resource in resources:
            if (
                resource.type == PrivateSearchResourceType.FOLDER
                and resource.child_ids
                and resource_id in resource.child_ids
            ):
                return resource.name
        return None

    @staticmethod
    def get_type(
        resource_id: str, resources: list[Resource]
    ) -> PrivateSearchResourceType:
        for resource in resources:
            if resource.id == resource_id:
                return resource.type
        # Chunk hits are documents; empty visible_resources used to yield None.
        return PrivateSearchResourceType.RESOURCE

    def get_function(
        self, private_search_tool: PrivateSearchTool, **kwargs
    ) -> SearchFunction:
        return partial(
            self.query, private_search_tool=private_search_tool, k=40, **kwargs
        )

    @classmethod
    def get_schema(cls) -> dict:
        return cls.generate_schema(
            "private_search",
            'Search for user\'s private & personal resources. Return in <cite id=""></cite> format.',
            display_name={"zh": "知识库搜索", "en": "Knowledge Base Search"},
        )

    @tracer.start_as_current_span("WeaviateVectorRetriever.query")
    async def query(
        self,
        query: str,
        k: int,
        *,
        private_search_tool: PrivateSearchTool,
        trace_info: TraceInfo | None = None,
    ) -> list[ResourceChunkRetrieval]:
        condition: Condition = private_search_tool.to_condition()
        recall_result_list = await self.vector_db.query_chunks(
            private_search_tool.namespace_id, query, k, condition
        )
        retrievals: List[ResourceChunkRetrieval] = [
            ResourceChunkRetrieval(
                chunk=chunk,
                folder=self.get_folder(
                    chunk.resource_id, private_search_tool.resources or []
                ),
                type=self.get_type(
                    chunk.resource_id, private_search_tool.visible_resources or []
                ),
                namespace_id=private_search_tool.namespace_id,
                score=Score(recall=score, rerank=0),
            )
            for chunk, score in recall_result_list
        ]
        trace_info and trace_info.debug(
            {
                "condition": condition.model_dump() if condition else condition,
                "len(retrievals)": len(retrievals),
            }
        )
        return retrievals
