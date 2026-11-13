import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

RESOURCE_HASH_NAMESPACE = "resource_hash"

_URL_PATTERN = re.compile(r"https?://[^\s\"'<>\\]+")
_ISO_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_CITATION_PREFIX_PATTERN = re.compile(r"\bC\d+-")
_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HISTORY_MESSAGE_PATTERN = re.compile(
    r'<message id="([^"]+)"[^>]*>(.*?)</message>', re.DOTALL
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def result_fingerprint(
    tool_name: str,
    args: dict | None,
    content: Any,
    attrs: dict | None = None,
) -> str:
    value, customized = _tool_fingerprint_value(tool_name, args, content, attrs)
    if not isinstance(value, str):
        value = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    normalized = strip_volatile(
        value, strip_timestamps=not customized, strip_uuids=not customized
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _tool_fingerprint_value(
    tool_name: str, args: dict | None, content: Any, attrs: dict | None
) -> tuple[Any, bool]:
    attrs = attrs or {}
    if tool_name == "read_resource":
        value = _resource_hash_operation(attrs, args)
        if value is not None:
            return value, True
    if tool_name == "list_resources":
        value = _list_resources_value(content)
        if value is not None:
            return value, True
    if tool_name == "read_root":
        value = _read_root_value(content)
        if value is not None:
            return value, True
    if tool_name == "describe_resource":
        value = _describe_resource_value(content)
        if value is not None:
            return value, True
    if tool_name in ("list_resource_attachments", "get_resource_attachment"):
        value = _attachments_value(content)
        if value is not None:
            return value, True
    if tool_name == "view_attachment":
        return content_text(content), True
    if tool_name in ("resource_search", "web_search", "search", "private_search"):
        citations = _citation_values(
            attrs, include_updated_at=tool_name == "resource_search"
        )
        if citations is not None:
            return citations, True
    if tool_name == "resource_filter":
        value = _resource_filter_value(content)
        if value is not None:
            return value, True
    if tool_name in ("list_tags", "tag_filter"):
        value = _parse_json_value(content)
        if value is not None:
            return value, True
    if tool_name == "list_resource_comments":
        value = _comments_value(content)
        if value is not None:
            return value, True
    if tool_name in (
        "get_rss_folder_config",
        "get_smart_folder_config",
        "get_current_info",
        "load_docs",
        "list_local_devices",
    ):
        value = _parse_json_value(content)
        if value is not None:
            return value, True
        return content_text(content), False
    if tool_name == "search_conversation_messages":
        value = _history_recall_value(content)
        if value is not None:
            return value, True
    if tool_name == "read_conversation_message":
        value = _conversation_message_value(content)
        if value is not None:
            return value, True
    return _generic_value(content), False


def _generic_value(content: Any) -> str:
    text = content_text(content).strip()
    canonical = _canonical_json(text)
    return canonical if canonical is not None else text


def _canonical_json(text: str) -> str | None:
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _parse_json_value(content: Any) -> Any | None:
    text = content_text(content).strip()
    canonical = _canonical_json(text)
    if canonical is None:
        return None
    return json.loads(canonical)


def _json_without_prefix(content: Any) -> Any | None:
    text = content_text(content).strip()
    lines = text.splitlines()
    while lines and lines[0].startswith("Only "):
        lines.pop(0)
    if not lines:
        return None
    try:
        return json.loads("\n".join(lines))
    except (TypeError, ValueError):
        return None


def _resource_hash_operation(attrs: dict, args: dict | None) -> str | None:
    resource_id = (args or {}).get("resource_id")
    context = attrs.get("context")
    if not isinstance(context, dict):
        return None
    operations = context.get("operations")
    if not isinstance(operations, list):
        return None
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        if (
            operation.get("namespace") == RESOURCE_HASH_NAMESPACE
            and operation.get("key") == resource_id
        ):
            value = operation.get("value")
            if value is not None:
                return str(value)
    return None


def _list_resources_value(content: Any) -> Any | None:
    parsed = _json_without_prefix(content)
    if not isinstance(parsed, dict):
        return None
    resources = parsed.get("resources")
    if not isinstance(resources, list):
        return None
    items = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        items.append(
            [
                resource.get("id"),
                resource.get("name"),
                resource.get("type") or resource.get("resource_type"),
                resource.get("has_children"),
            ]
        )
    items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return {"resources": items, "total": parsed.get("total")}


def _read_root_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    candidates: list[dict] = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("roots"), list):
            candidates = [root for root in parsed["roots"] if isinstance(root, dict)]
        else:
            candidates = [root for root in parsed.values() if isinstance(root, dict)]
    elif isinstance(parsed, list):
        candidates = [root for root in parsed if isinstance(root, dict)]
    if not candidates:
        return None
    items = [[root.get("id"), root.get("name")] for root in candidates]
    items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return items


def _describe_resource_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    if not isinstance(parsed, dict):
        return None
    return [
        parsed.get("id"),
        parsed.get("updated_at"),
        parsed.get("sub_resource_count"),
        parsed.get("content_line_count"),
        parsed.get("tags"),
    ]


def _attachments_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    attachments: Any = None
    if isinstance(parsed, dict):
        attachments = parsed.get("attachments")
        if attachments is None:
            attachment = parsed.get("attachment")
            if isinstance(attachment, dict):
                attachments = [attachment]
            elif "id" in parsed and "name" in parsed and "size" in parsed:
                attachments = [parsed]
    elif isinstance(parsed, list):
        attachments = parsed
    if not isinstance(attachments, list):
        return None
    items = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        items.append(
            [
                attachment.get("id"),
                attachment.get("name"),
                attachment.get("size"),
            ]
        )
    items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return items


def _citation_values(attrs: dict, *, include_updated_at: bool) -> list | None:
    citations = attrs.get("citations")
    if not isinstance(citations, list):
        return None
    values = []
    for citation in citations:
        values.append(_citation_key(citation, include_updated_at))
    values.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return values


def _citation_key(citation: Any, include_updated_at: bool) -> Any:
    if hasattr(citation, "model_dump"):
        citation = citation.model_dump(mode="json")
    if not isinstance(citation, dict):
        return str(citation)
    citation_id = citation.get("id")
    if isinstance(citation_id, str):
        citation_id = _CITATION_PREFIX_PATTERN.sub("", citation_id)
    key = [
        citation.get("link"),
        citation.get("title"),
        citation.get("snippet"),
        citation_id,
    ]
    if include_updated_at:
        key.append(citation.get("updated_at"))
    return key


def _resource_filter_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    if not isinstance(parsed, dict):
        return None
    resources = parsed.get("resources")
    if not isinstance(resources, list):
        return None
    items = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        items.append([resource.get("id"), resource.get("updated_at")])
    items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return items


def _comments_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    if not isinstance(parsed, dict):
        return None
    threads = parsed.get("threads")
    if not isinstance(threads, list):
        return None
    items = []
    for thread in threads:
        if not isinstance(thread, dict):
            continue
        comments = []
        for comment in thread.get("comments") or []:
            if isinstance(comment, dict):
                comments.append([comment.get("id"), comment.get("updated_at")])
        items.append([thread.get("id"), comments])
    items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return items


def _conversation_message_value(content: Any) -> Any | None:
    parsed = _parse_json_value(content)
    if not isinstance(parsed, dict):
        return None
    payload = parsed.get("message")
    if not isinstance(payload, dict):
        payload = parsed
    return {
        "id": payload.get("id") or payload.get("message_id"),
        "content": payload.get("content"),
    }


def _history_recall_value(content: Any) -> Any | None:
    text = content_text(content)
    if "<history_recall>" not in text:
        return None
    hits = []
    for message_id, body in _HISTORY_MESSAGE_PATTERN.findall(text):
        digest = hashlib.sha256(body.strip().encode("utf-8")).hexdigest()
        hits.append([message_id, digest])
    if not hits:
        return None
    hits.sort(key=lambda item: json.dumps(item, ensure_ascii=False, default=str))
    return hits


def strip_volatile(
    text: str, *, strip_timestamps: bool = True, strip_uuids: bool = True
) -> str:
    canonical = _canonical_json(text.strip())
    if canonical is not None:
        text = canonical
    text = _URL_PATTERN.sub(_strip_url_volatile, text)
    if strip_timestamps:
        text = _ISO_TIMESTAMP_PATTERN.sub("<ts>", text)
    text = _CITATION_PREFIX_PATTERN.sub("", text)
    if strip_uuids:
        text = _UUID_PATTERN.sub("<id>", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _strip_url_volatile(match: re.Match[str]) -> str:
    url = match.group(0)
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
