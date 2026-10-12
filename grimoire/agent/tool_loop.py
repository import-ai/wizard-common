import json
from importlib.resources import files

from common.template_parser import TemplateParser

MODEL_TOOL_STOP_NOTE = (
    "Same arguments were already used, or the tool round limit was reached. "
    "Do not call this tool again. Answer the user in text."
)


def canonical_tool_key(
    name: str, arguments: str | dict | list | None
) -> tuple[str, str]:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return name, arguments
        arguments = parsed
    if isinstance(arguments, (dict, list)):
        return name, json.dumps(
            arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    return name, "" if arguments is None else str(arguments)


def render_tool_loop_fallback(lang: str | None) -> str:
    parser = TemplateParser(
        base_dir=str(files("wizard_common") / "resources" / "prompt_templates")
    )
    template = parser.get_template("tool_loop_fallback.j2")
    return parser.render_template(template, lang=lang or "简体中文")
