import asyncio
from functools import partial
from typing import Any, List, Tuple

from common.trace_info import TraceInfo
from openai import AsyncOpenAI
from opentelemetry import propagate, trace
from wizard_common.grimoire.config import VectorConfig
from wizard_common.grimoire.entity.chunk import Chunk, ResourceChunkRetrieval
from wizard_common.grimoire.entity.index_record import IndexRecord, IndexRecordType
from wizard_common.grimoire.entity.message import Message
from wizard_common.grimoire.entity.retrieval import Score
from wizard_common.grimoire.entity.tools import (
    Condition,
    PrivateSearchResourceType,
    PrivateSearchTool,
    Resource,
)
from wizard_common.grimoire.retriever.base import BaseRetriever, SearchFunction

import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateDeleteManyError

tracer = trace.get_tracer(__name__)
COLLECTION_NAME = "omnibox_index"


class WeaviateVectorDB:
    def __init__(self, config: VectorConfig):
        self.config: VectorConfig = config
        self.batch_size: int = config.batch_size
        self.openai = AsyncOpenAI(
            api_key=config.embedding.api_key, base_url=config.embedding.base_url
        )
        self.client: weaviate.WeaviateAsyncClient = ...
        self._init_lock = asyncio.Lock()
        self.dimension = config.dimension

    async def _ensure_client(self) -> None:
        if self.client is not ...:
            return
        async with self._init_lock:
            if self.client is not ...:
                return

            connect_kwargs: dict[str, Any] = {"port": self.config.weaviate.port}
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
                name="chunk_title",
                data_type=wvc.config.DataType.TEXT,
                index_searchable=True,
                tokenization=wvc.config.Tokenization.GSE_CH,
            ),
            wvc.config.Property(
                name="chunk_text",
                data_type=wvc.config.DataType.TEXT,
                index_searchable=True,
                tokenization=wvc.config.Tokenization.GSE_CH,
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
                name="message_content",
                data_type=wvc.config.DataType.TEXT,
                index_searchable=True,
                tokenization=wvc.config.Tokenization.GSE_CH,
            ),
        ]

    async def _ensure_collection_properties(self) -> None:
        required_properties = self._required_collection_properties()
        collection = self.client.collections.get(COLLECTION_NAME)
        config = await collection.config.get()
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
                index_null_state=True
            ),
            properties=required_properties,
        )

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

    async def _hybrid_query(
        self,
        namespace_id: str,
        query: str,
        condition: Condition,
        limit: int,
        offset: int = 0,
    ) -> List[Tuple[dict, float]]:
        collection = await self._get_shard(namespace_id)
        vector = (await self._embed(query))[0] if query else None

        search_limit = limit + offset
        try:
            response = await collection.query.hybrid(
                query=query or "",
                vector=vector,
                alpha=0.5,
                filters=condition.to_weaviate_filters(),
                limit=search_limit,
                return_metadata=wvc.query.MetadataQuery.full(),
            )
        except UnexpectedStatusCodeError as e:
            # Tenant not found -> no data yet.
            if e.status_code == 422:
                return []
            raise

        hits: list[Tuple[dict, float]] = []
        for obj in response.objects:
            properties = obj.properties or {}
            score = 0.0
            if obj.metadata and obj.metadata.score is not None:
                score = float(obj.metadata.score)
            elif obj.metadata and obj.metadata.certainty is not None:
                score = float(obj.metadata.certainty)
            hits.append((properties, score))

        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[offset : offset + limit]

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
                properties["chunk_text"] = chunk.text
                properties["chunk_resource_id"] = chunk.resource_id
                properties["chunk_parent_id"] = chunk.parent_id
                properties["chunk_type"] = chunk.chunk_type.value
                properties["chunk_id"] = chunk.chunk_id
                properties["chunk_created_at"] = chunk.created_at
                properties["chunk_updated_at"] = chunk.updated_at
                properties["chunk_start_index"] = chunk.start_index
                properties["chunk_end_index"] = chunk.end_index
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

        try:
            await collection.data.delete_many(
                where=wvc.query.Filter.by_property("message_id").equal(
                    message.message_id
                )
            )
        except WeaviateDeleteManyError:
            # Tenant not found (no data yet for this namespace)
            pass

        message_content = message.message.content.strip()
        if not message_content:
            return

        vector = (await self._embed(message_content))[0]
        properties = {
            "type": IndexRecordType.message.value,
            "namespace_id": namespace_id,
            "user_id": user_id,
        }
        properties["message_id"] = message.message_id
        properties["conversation_id"] = message.conversation_id
        properties["message_role"] = message.message.role
        properties["message_content"] = message_content

        await collection.data.insert(properties=properties, vector=vector)

    @tracer.start_as_current_span("WeaviateVectorDB.remove_conversation")
    async def remove_conversation(self, namespace_id: str, conversation_id: str):
        collection = await self._get_shard(namespace_id)
        try:
            ret = await collection.data.delete_many(
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
            ret = await collection.data.delete_many(
                where=wvc.query.Filter.by_property("type").equal(
                    IndexRecordType.chunk.value
                )
                & wvc.query.Filter.by_property("namespace_id").equal(namespace_id)
                & wvc.query.Filter.by_property("chunk_resource_id").equal(resource_id)
            )
        except WeaviateDeleteManyError:
            # Tenant not found (no data yet for this namespace)
            pass

    @tracer.start_as_current_span("WeaviateVectorDB.search")
    async def search(
        self,
        query: str,
        namespace_id: str,
        user_id: str | None,
        record_type: IndexRecordType | None,
        offset: int,
        limit: int,
    ) -> List[IndexRecord]:
        condition = Condition(
            namespace_id=namespace_id,
            user_id=user_id,
            record_type=record_type.value if record_type else None,
        )

        hits = await self._hybrid_query(
            namespace_id=namespace_id,
            query=query,
            condition=condition,
            limit=limit,
            offset=offset,
        )
        return [IndexRecord(**hit) for hit, _ in hits]

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
            chunk = Chunk(
                title=hit.get("chunk_title"),
                resource_id=hit["chunk_resource_id"],
                text=hit.get("chunk_text"),
                chunk_type=hit["chunk_type"],
                parent_id=hit["chunk_parent_id"],
                chunk_id=hit["chunk_id"],
                created_at=hit["chunk_created_at"],
                updated_at=hit["chunk_updated_at"],
                start_index=hit.get("chunk_start_index"),
                end_index=hit.get("chunk_end_index"),
            )
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
    ) -> PrivateSearchResourceType | None:
        for resource in resources:
            if resource.id == resource_id:
                return resource.type
        return None

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
                "where": condition.to_weaviate_filters(),
                "condition": condition.model_dump() if condition else condition,
                "len(retrievals)": len(retrievals),
            }
        )
        return retrievals
