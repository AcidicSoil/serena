from serena.config.context_mode import SerenaAgentContext

CHATGPT_ACTION_DESCRIPTION_MAX_CHARS = 300
CHATGPT_ACTION_DESCRIPTION_HEADROOM_MAX_CHARS = 180
REQUIRED_SHORT_DESCRIPTION_OVERRIDES = {
    "replace_in_files",
    "search_for_pattern",
    "write_memory",
    "execute_shell_command",
}
HEADROOM_DESCRIPTION_OVERRIDES = {
    "replace_in_files",
    "execute_shell_command",
}


def test_chatgpt_tool_description_overrides_fit_actions_limit() -> None:
    context = SerenaAgentContext.from_name("chatgpt")

    assert REQUIRED_SHORT_DESCRIPTION_OVERRIDES <= context.tool_description_overrides.keys()

    descriptions_over_limit = {
        tool_name: len(description)
        for tool_name, description in context.tool_description_overrides.items()
        if len(description) > CHATGPT_ACTION_DESCRIPTION_MAX_CHARS
    }
    assert descriptions_over_limit == {}


def test_chatgpt_tool_descriptions_leave_schema_headroom() -> None:
    context = SerenaAgentContext.from_name("chatgpt")

    descriptions_over_headroom_budget = {
        tool_name: len(context.tool_description_overrides[tool_name])
        for tool_name in HEADROOM_DESCRIPTION_OVERRIDES
        if len(context.tool_description_overrides[tool_name]) > CHATGPT_ACTION_DESCRIPTION_HEADROOM_MAX_CHARS
    }
    assert descriptions_over_headroom_budget == {}
