from dataclasses import dataclass, field


@dataclass
class ChatOptions:
    enable_thinking: bool = False
    enable_search: bool = False


@dataclass
class ChatResult:
    content: str
    reasoning: str | None = None
    search_queries: list[str] = field(default_factory=list)
    search_status: str = "none"  # none | pending | success | failed | timeout


@dataclass
class ProviderCapabilities:
    supports_thinking: bool = False
    thinking_toggle: bool = False
    supports_native_search: bool = False
    supports_tool_search: bool = False
    thinking_search_exclusive: bool = False

    def to_dict(self) -> dict:
        return {
            "supports_thinking": self.supports_thinking,
            "thinking_toggle": self.thinking_toggle,
            "supports_native_search": self.supports_native_search,
            "supports_tool_search": self.supports_tool_search,
            "thinking_search_exclusive": self.thinking_search_exclusive,
        }
