"""Tools for opening and closing CPACS sessions."""

from __future__ import annotations

import pathlib
from importlib import import_module
from typing import Any, Literal, cast

from tigl_mcp import cpacs_stubs
from tigl_mcp.cpacs import build_handles, parse_cpacs
from tigl_mcp.errors import MCPError, raise_mcp_error
from tigl_mcp.session_manager import SessionManager
from tigl_mcp.tooling import ToolDefinition, ToolParameters
from tigl_mcp.tools.common import require_session


class OpenCpacsParams(ToolParameters):
    """Parameters for the open_cpacs tool."""

    source_type: Literal["path", "xml_string"]
    source: str


class CloseCpacsParams(ToolParameters):
    """Parameters for the close_cpacs tool."""

    session_id: str


class ExportCpacsParams(ToolParameters):
    """Parameters for the export_cpacs tool."""

    session_id: str
    output_path: str


def _read_source(params: OpenCpacsParams) -> tuple[str, str | None]:
    if params.source_type == "path":
        path = pathlib.Path(params.source)
        if not path.exists():
            raise_mcp_error("InvalidInput", f"File not found: {path}")
        return path.read_text(encoding="utf-8"), str(path)
    return params.source, None


def _load_real_bindings() -> tuple[object | None, object | None]:
    """Load TiXI/TiGL wrappers when optional native dependencies are present."""
    try:
        tigl_wrapper = import_module("tigl3.tigl3wrapper")
        tixi_wrapper = import_module("tixi3.tixi3wrapper")
        return tixi_wrapper, tigl_wrapper
    except Exception:  # pragma: no cover - optional runtime dependency
        return None, None


def open_cpacs_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the open_cpacs tool definition."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = OpenCpacsParams.model_validate(raw_params)
            xml_content, file_name = _read_source(params)
            cpacs_config = parse_cpacs(xml_content)
            tixi3wrapper, tigl3wrapper = _load_real_bindings()

            if (
                tixi3wrapper is not None and tigl3wrapper is not None
            ):  # pragma: no cover
                tixi_module = cast(Any, tixi3wrapper)
                tigl_module = cast(Any, tigl3wrapper)

                real_tixi: Any = tixi_module.Tixi3()
                if hasattr(real_tixi, "openString"):
                    real_tixi.openString(xml_content)
                elif hasattr(real_tixi, "openDocumentFromString"):
                    real_tixi.openDocumentFromString(xml_content)
                else:
                    raise_mcp_error(
                        "OpenError",
                        "No supported TIXI open-from-string API found.",
                    )

                real_tigl: Any = tigl_module.Tigl3()
                real_tigl.open(real_tixi, "")

                # Wrap the real handles in our dataclass wrappers so the rest
                # of the server can access both parsed config and native API.
                from tigl_mcp.cpacs import TiglConfiguration, TixiDocument

                tixi_handle: Any = TixiDocument(
                    xml_content=xml_content,
                    file_name=file_name,
                    _tixi_handle=real_tixi,
                )
                tigl_handle: Any = TiglConfiguration(
                    cpacs_configuration=cpacs_config,
                    _tigl_handle=real_tigl,
                )
            else:
                tixi_handle, tigl_handle, _, _ = build_handles(xml_content, file_name)

            session_id = session_manager.create_session(
                tixi_handle, tigl_handle, cpacs_config, xml_content
            )
            summary = {
                "num_wings": len(cpacs_config.wings),
                "num_fuselages": len(cpacs_config.fuselages),
                "num_rotors": len(cpacs_config.rotors),
                "num_engines": len(cpacs_config.engines),
            }
            return {
                "session_id": session_id,
                "cpacs_metadata": cpacs_stubs.extract_metadata(xml_content, file_name),
                "configuration_summary": summary,
            }
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error("OpenError", "Failed to open CPACS", str(exc))

    return ToolDefinition(
        name="open_cpacs",
        description="Open a CPACS session from a file path or an XML string.",
        parameters_model=OpenCpacsParams,
        handler=handler,
        output_schema={},
    )


def close_cpacs_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the close_cpacs tool definition."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = CloseCpacsParams.model_validate(raw_params)
            # AUTO-EXPORT the (possibly morphed) geometry to an absolute path BEFORE
            # closing, so disciplines that read CPACS from disk (mass-mcp) always see
            # the current design — independent of whether the agent called export_cpacs
            # or whether export_cpacs is in its toolset. This makes the structures
            # coupling work uniformly across ALL coordination combinations. Best-effort:
            # never let an export problem block the close.
            auto_path: str | None = None
            export_error: str = ""
            try:
                tixi_handle, _tigl, _cfg = require_session(
                    session_manager, params.session_id
                )
                native = getattr(tixi_handle, "_tixi_handle", None)
                if native is not None and hasattr(native, "exportDocumentAsString"):
                    out = pathlib.Path(
                        f"/tmp/cpacs_autoexport_{params.session_id}.xml"
                    )
                    out.write_text(
                        native.exportDocumentAsString(), encoding="utf-8"
                    )
                    if out.is_file() and out.stat().st_size > 0:
                        auto_path = str(out)
            except Exception as exc:  # noqa: BLE001 - export is best-effort
                auto_path = None
                export_error = str(exc)
            session_manager.close(params.session_id)
            resp: dict[str, object] = {"success": True}
            if auto_path:
                resp["cpacs_file_path"] = auto_path
                resp["auto_exported"] = True
            else:
                # The close genuinely succeeded, so `success` stays True -- but a
                # silent auto-export failure is not harmless. This file is what
                # downstream mass sizing reads to see the MORPHED geometry; with
                # no file and no stated reason, the caller falls back to the
                # original CPACS path and sizes the BASELINE aircraft while
                # everything still reports success (the open B21 symptom).
                resp["auto_exported"] = False
                resp["auto_export_error"] = export_error or (
                    "this session exposes no native TiXI handle, so the morphed "
                    "geometry could not be written"
                )
                resp["warning"] = (
                    "No morphed CPACS file was exported. Anything downstream that "
                    "reads geometry will see the ORIGINAL file, not this session's "
                    "changes."
                )
            return resp
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error("CloseError", "Failed to close CPACS", str(exc))

    return ToolDefinition(
        name="close_cpacs",
        description=(
            "Close a CPACS session and free resources. Auto-exports the current "
            "(possibly morphed) geometry to a file first so downstream mass sizing "
            "reads the actual design."
        ),
        parameters_model=CloseCpacsParams,
        handler=handler,
        output_schema={},
    )


def export_cpacs_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the export_cpacs tool definition.

    Persists a session's CURRENT (possibly morphed) CPACS geometry to a file.
    ``set_high_level_parameters`` / ``morph_wing`` / ``morph_fuselage`` write the
    deformation back into the session's live TiXI document (WriteCPACS), so this
    tool exports that document — letting downstream disciplines that read CPACS
    from disk (e.g. mass-mcp ``estimate_mass``) see the MORPHED design rather than
    the original baseline file. Without it, structural mass is computed on baseline
    geometry regardless of the morph (geometry-decoupled).
    """

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = ExportCpacsParams.model_validate(raw_params)
            tixi_handle, _tigl_handle, _config = require_session(
                session_manager, params.session_id
            )
            # Resolve to an ABSOLUTE path: each MCP server has its own working
            # directory, so a relative output_path would land under the tigl
            # server's cwd and be invisible to the mass discipline. Returning an
            # absolute path makes the export unambiguous for every downstream reader.
            out = pathlib.Path(params.output_path).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)

            native = getattr(tixi_handle, "_tixi_handle", None)
            if native is not None and hasattr(native, "exportDocumentAsString"):
                # Most deterministic: serialize the live in-memory tree (includes
                # every morph edit) and write it ourselves.
                xml = native.exportDocumentAsString()
                out.write_text(xml, encoding="utf-8")
                source = "tixi-native (morphed geometry)"
            elif native is not None and hasattr(native, "saveDocument"):
                native.saveDocument(str(out))
                source = "tixi-native saveDocument (morphed geometry)"
            else:
                # Stub / no native runtime — best effort: last known XML string.
                out.write_text(
                    getattr(tixi_handle, "xml_content", "") or "", encoding="utf-8"
                )
                source = "xml_content (stub — no native TiXI runtime)"

            if not out.exists() or out.stat().st_size == 0:
                raise_mcp_error(
                    "ExportError",
                    f"CPACS export produced no content at {out}",
                )
            return {
                "status": "success",
                "cpacs_file_path": str(out),
                "source": source,
            }
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error("ExportError", "Failed to export CPACS", str(exc))

    return ToolDefinition(
        name="export_cpacs",
        description=(
            "Save the current (morphed) CPACS geometry of a session to a file so "
            "disk-reading disciplines (e.g. mass estimation) see the morphed design."
        ),
        parameters_model=ExportCpacsParams,
        handler=handler,
        output_schema={},
    )
