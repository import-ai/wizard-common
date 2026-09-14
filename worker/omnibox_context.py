import httpx

_LIST_PAGE_LIMIT = 200
_OMNIBOX_DIR = ".omnibox"
_SPACES = ("private", "teamspace")


async def _find_child_id(
    client: httpx.AsyncClient, parent_id: str, name: str
) -> str | None:
    offset = 0
    while True:
        response = await client.get(
            f"/resources/{parent_id}/list",
            params={"offset": offset, "limit": _LIST_PAGE_LIMIT},
        )
        response.raise_for_status()
        data = response.json()
        resources = data.get("resources") or []
        for resource in resources:
            if resource.get("name") == name:
                return resource.get("id")
        offset += len(resources)
        total = data.get("total") or 0
        if offset >= total or not resources:
            return None


async def _read_space_file(
    client: httpx.AsyncClient, root_id: str, filename: str
) -> str | None:
    omnibox_id = await _find_child_id(client, root_id, _OMNIBOX_DIR)
    if not omnibox_id:
        return None
    file_id = await _find_child_id(client, omnibox_id, filename)
    if not file_id:
        return None
    response = await client.get(f"/resources/{file_id}")
    response.raise_for_status()
    content = response.json().get("content")
    if not content or not str(content).strip():
        return None
    return str(content)


async def _resolve_space(
    client: httpx.AsyncClient, resource_id: str
) -> str | None:
    """Which space the resource being processed lives in, or None if unknown."""
    response = await client.get(f"/resources/{resource_id}")
    response.raise_for_status()
    space = response.json().get("space_type")
    return space if space in _SPACES else None


async def load_omnibox_markdown(
    *,
    filename: str,
    base_url: str | None = None,
    namespace_id: str | None = None,
    user_id: str | None = None,
    resource_id: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Load `.omnibox/<filename>` for the space `resource_id` lives in.

    A private resource reads the private file and a teamspace resource reads the
    teamspace one, so a team's rules do not leak into someone's private notes
    and vice versa. Both spaces are read when the resource's space cannot be
    determined, which keeps the file useful rather than silently dropping it.

    Failures return an empty string so callers can keep using the default
    prompt. Missing files are skipped and never created.
    """
    close_client = False
    http_client = client
    try:
        if http_client is None:
            if not base_url or not namespace_id or not user_id:
                return ""
            http_client = httpx.AsyncClient(
                base_url=f"{base_url.rstrip('/')}/internal/api/v1/namespaces/{namespace_id}",
                headers={"X-User-ID": user_id},
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
            close_client = True
        roots_response = await http_client.get("/roots")
        roots_response.raise_for_status()
        roots = roots_response.json()
    except Exception:
        if close_client and http_client is not None:
            await http_client.aclose()
        return ""

    spaces = _SPACES
    if resource_id:
        try:
            resolved = await _resolve_space(http_client, resource_id)
            if resolved:
                spaces = (resolved,)
        except Exception:
            pass

    sections: list[str] = []
    try:
        for space in spaces:
            try:
                root_id = (roots.get(space) or {}).get("id")
                if not root_id:
                    continue
                content = await _read_space_file(http_client, root_id, filename)
                if content:
                    sections.append(f"# /{space}/.omnibox/{filename}\n{content}")
            except Exception:
                continue
    finally:
        if close_client:
            await http_client.aclose()
    return "\n\n".join(sections)
