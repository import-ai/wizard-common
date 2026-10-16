from wizard_common.grimoire.agent.saved_resource import saved_resource_prompt_content


def test_saved_resource_prompt_contains_escaped_raw_content_and_metadata():
    prompt = saved_resource_prompt_content(
        "2 < 3 & 4",
        {
            "channel": "qq",
            "saved_resource": {
                "resource_id": "res-1",
                "resource_type": "doc",
                "created_at": "2026-10-14T08:00:00.000Z",
            },
        },
    )

    assert prompt is not None
    assert "<channel>qq</channel>" in prompt
    assert "<resource_id>res-1</resource_id>" in prompt
    assert "<content>2 &lt; 3 &amp; 4</content>" in prompt


def test_non_saved_messages_are_unchanged():
    assert saved_resource_prompt_content("hello", {}) is None
