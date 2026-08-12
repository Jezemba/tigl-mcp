"""Custom error types for MCP tooling."""

from __future__ import annotations

from typing import NoReturn, TypedDict


class MCPErrorPayload(TypedDict):
    """Structured JSON payload for MCP errors."""

    error: dict[str, object | None]


class MCPError(Exception):
    """Structured MCP error containing a JSON-friendly payload."""

    def __init__(
        self, error_type: str, message: str, details: object | None = None
    ) -> None:
        """Create a structured MCP error payload."""
        super().__init__(message)
        self.error: MCPErrorPayload = {
            "error": {
                "type": error_type,
                "message": message,
                "details": details,
            }
        }

    def to_dict(self) -> MCPErrorPayload:
        """Return the structured error payload."""
        return self.error


def raise_mcp_error(
    error_type: str, message: str, details: object | None = None
) -> NoReturn:
    """Raise an :class:`MCPError` with a structured payload."""
    raise MCPError(error_type=error_type, message=message, details=details)


def raise_component_not_found(config: object, uid: str, kind: str = "Component") -> NoReturn:
    """Report an unknown component UID *with the UIDs that do exist*.

    ``Component 'D150_Wing' not found`` was the single worst error in the system:
    ``set_high_level_parameters`` had a **50% identical-retry rate** (25 of 50
    calls across the 2026-08 sweeps were byte-identical repeats). The observed
    sequence is a caller guessing names one at a time --

        D150_Wing -> not found      (~91 s)
        Wing_1    -> not found      (~80 s)
        Fuselage  -> not found      (~57 s)

    -- while the real UIDs are ``Wing1`` and ``Fuselage1``. The session has the
    CPACS file open and ``config.all_components()`` is one call away, so the
    server knew the answer at every one of those steps and withheld it. Naming
    the candidates turns an unbounded guessing loop into a single correction.
    """
    try:
        available = sorted(c.uid for c in config.all_components())
    except Exception:  # pragma: no cover - never fail while reporting a failure
        available = []

    if not available:
        raise_mcp_error(
            "NotFound",
            f"{kind} '{uid}' not found, and this configuration exposes no components. "
            "Open a CPACS file for this session first.",
        )

    import difflib

    near = difflib.get_close_matches(uid, available, n=1, cutoff=0.6)
    suggestion = f" Did you mean '{near[0]}'?" if near else ""
    raise_mcp_error(
        "NotFound",
        f"{kind} '{uid}' not found.{suggestion} Available UIDs: "
        f"{', '.join(available)}. Use one of these exactly (they are the UIDs "
        "defined in the CPACS file, not display names).",
        details={"requested": uid, "available_uids": available},
    )
