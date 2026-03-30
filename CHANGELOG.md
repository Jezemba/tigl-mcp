# Changelog

## [Unreleased]

### Added
- Real TiGL/TiXI backend support: when `tigl3` and `tixi3` are installed the
  server now uses native API calls instead of stub/synthetic data.
- `_tigl_handle` and `_tixi_handle` fields on `TiglConfiguration` and
  `TixiDocument` dataclasses to hold optional native handles.
- `open_cpacs` wraps real TiGL/TiXI handles inside the existing dataclass
  wrappers so all downstream tools can access both parsed config and native API.
- `get_wing_summary` and `get_fuselage_summary` use real TiGL geometry queries
  (`wingGetSpan`, `wingGetReferenceArea`, `wingGetMAC`, `fuselageGetVolume`,
  etc.) when available, falling back to stub calculations.
- `export_component_mesh` and `export_configuration_cad` route through the
  native TiGL handle for STL, STEP, and IGES export when available.
- New `generate_volume_mesh` tool: creates CFD-ready 3D volume meshes using
  gmsh with proper far-field domain, surface embedding, boundary markers
  ("aircraft" / "farfield"), and SU2 or MSH output.
- `geometry` optional-dependency group in `pyproject.toml` for `tigl3`, `tixi3`,
  and `gmsh`.

### Changed
- The server continues to work in stub mode when `tigl3` is not installed;
  no behavioral change for existing deployments.
