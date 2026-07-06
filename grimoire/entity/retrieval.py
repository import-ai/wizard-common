from abc import abstractmethod, ABC
import re
import unicodedata
from urllib.parse import urlparse

from anyascii import anyascii
from pydantic import BaseModel, Field, field_validator
from pypinyin import lazy_pinyin

from common.utils import remove_continuous_break_lines


CITATION_ID_PATTERN = re.compile(r"^C(\d+)(?:-|$)")
HAN_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def transliterate_to_ascii(text: str) -> str:
    def replace_han(match: re.Match[str]) -> str:
        return " " + " ".join(lazy_pinyin(match.group(0), errors="ignore")) + " "

    text = HAN_PATTERN.sub(replace_han, text)
    return anyascii(text)


def get_domain(url: str) -> str:
    return urlparse(url).netloc


def make_citation_slug(
    title: str | None,
    snippet: str | None,
    *,
    max_tokens: int = 3,
    max_length: int = 40,
) -> str:
    text = unicodedata.normalize("NFKC", f"{title or ''} {snippet or ''}")
    text = transliterate_to_ascii(text).lower()
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text):
        if token.isdigit() or len(token) <= 1 or token in tokens:
            continue
        tokens.append(token)
    if not tokens:
        return "source"
    slug = "-".join(tokens[:max_tokens])[:max_length].strip("-")
    return slug or "source"


def char_range_to_line_range(
    content: str, start_index: int | None, end_index: int | None
) -> dict[str, int] | None:
    if start_index is None or end_index is None:
        return None

    content_length = len(content)
    start = min(max(start_index, 0), content_length)
    end = min(max(end_index, start), content_length)
    end_for_line = max(start, end - 1)

    start_line = content.count("\n", 0, start) + 1
    end_line = content.count("\n", 0, end_for_line) + 1
    return {"start": start_line, "end": end_line}


def format_line_range(line_range: dict[str, int] | None) -> str | None:
    if line_range is None:
        return None
    return f"{line_range['start']}-{line_range['end']}"


def make_citation_id(
    index: int,
    title: str | None,
    snippet: str | None,
    line_range: str | None = None,
) -> str:
    citation_id = f"C{index}-{make_citation_slug(title, snippet)}"
    if line_range:
        citation_id = f"{citation_id}-L{line_range}"
    return citation_id


def format_cite_marker(citation_id: str) -> str:
    match = CITATION_ID_PATTERN.match(citation_id)
    if not match:
        return ""
    return f"[[{match.group(1)}]]({citation_id})"


class PromptContext(BaseModel, ABC):
    @abstractmethod
    def to_prompt(self) -> str:
        raise NotImplementedError("Subclasses should implement this method.")


def to_prompt(
    tag_attrs: dict, body_attrs: dict, i: str | None = None, tag_name: str = "cite"
) -> str:
    if i:
        tag_attrs = {"id": str(i)} | tag_attrs
    header_attrs: str = " ".join([f'{k}="{v}"' for k, v in tag_attrs.items() if v])
    contents: list[str] = [
        f"<{tag_name}{' ' if header_attrs else ''}{header_attrs}>",
        *[f"<{k}>{v}</{k}>" for k, v in body_attrs.items() if v],
        f"</{tag_name}>",
    ]
    return remove_continuous_break_lines("\n".join(contents))


class PromptCite(PromptContext, ABC):
    id: str = Field(default="")

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, value) -> str:
        if value is None:
            return ""
        citation_id = str(value)
        if citation_id.lstrip("-").isdigit():
            return ""
        return citation_id

    @abstractmethod
    def to_prompt(self, exclude_id: bool = False) -> str:
        raise NotImplementedError("Subclasses should implement this method.")


class Citation(PromptCite):
    title: str | None = None
    snippet: str | None = None
    link: str
    updated_at: str | None = None
    source: str | None = None
    namespace_id: str | None = None

    def to_prompt(self, exclude_id: bool = False) -> str:
        attrs: dict = self.model_dump(
            exclude_none=True, exclude={"snippet", "link", "id"}
        )
        cite_marker = "" if exclude_id else format_cite_marker(self.id)
        if cite_marker:
            attrs["cite_marker"] = cite_marker
        if (
            self.link
            and self.link.startswith("http")
            and (host := get_domain(self.link))
        ):
            attrs["host"] = host
        return to_prompt(
            attrs,
            self.model_dump(exclude_none=True, include={"snippet"}),
            i=self.id if cite_marker else None,
        )


class Score(BaseModel):
    recall: float | None = Field(default=None)
    rerank: float | None = Field(default=None)


class BaseRetrieval(PromptCite):
    score: Score = Field(default_factory=Score)
    source: str

    @abstractmethod
    def to_citation(self) -> Citation:
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.to_prompt(exclude_id=True) == other.to_prompt(exclude_id=True)
        return False


def retrievals2prompt(retrievals: list[PromptCite]) -> str:
    retrieval_prompt_list: list[str] = []
    for retrieval in retrievals:
        retrieval_prompt_list.append(retrieval.to_prompt())
    if retrieval_prompt_list:
        retrieval_prompt: str = "\n\n".join(retrieval_prompt_list)
    else:
        retrieval_prompt: str = ""
    return "\n".join(["<retrievals>", retrieval_prompt, "</retrievals>"])
