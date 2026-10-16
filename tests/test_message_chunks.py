from wizard_common.grimoire.retriever.weaviate_vector_db import split_message_content


def test_split_message_content_preserves_offsets_and_paragraphs():
    content = "first paragraph\n\nsecond paragraph"

    chunks = split_message_content(content, chunk_size=7)

    assert [text for text, _, _ in chunks] == [
        "first p",
        "aragrap",
        "h",
        "second ",
        "paragra",
        "ph",
    ]
    assert all(content[start:end] == text for text, start, end in chunks)
