# Changelog

## [Unreleased]

### Changed
- `generate_volume_mesh` now auto-scales mesh sizing RELATIVE to the geometry's
  characteristic length when `mesh_size_min`/`mesh_size_max` are left unset
  (now the default). This keeps cell count ~constant (~0.6-0.7M) whether a wing
  is morphed larger or smaller, instead of a fixed absolute size that ballooned
  to millions of cells (and timed out SU2) on an enlarged wing. Explicit sizes
  still override. Far-field distance was already a multiple of characteristic
  length. `surface_mesh_size` is accepted for compatibility but unused.
- `set_high_level_parameters` now actually DEFORMS wing/fuselage geometry (when a
  real TiGL runtime is present) instead of only recording intent in a dict. It
  translates the recorded parameters into a morph + rebuild (wing: area/AR from
  span + root/tip chord, plus sweep; fuselage: length + diameter) via the shared
  morph functions, and returns the measured new geometry under `geometry_morph`.
  This closes the aero↔geometry loop with NO change to the agents/prompts — they
  already call this tool. Falls back to record-only in stub mode.

### Added
- New `morph_wing` tool — closes the aero↔geometry coupling. Unlike
  `set_high_level_parameters` (which only records intent in an in-memory dict and
  never touches geometry), `morph_wing` deforms the live TiGL wing to hit a
  target reference area, aspect ratio, and/or sweep, then rebuilds so
  `generate_volume_mesh` / `exportWingBREPByUID` and downstream SU2 CFD reflect
  the new shape. Mechanism: fetch the live `CCPACSConfiguration` from the
  session handle, scale the wing transformation (x=chord, y=span, z=thickness;
  span ∝ y, reference_area ∝ x·y — verified exact on the D150) and set positioning
  sweep, `WriteCPACS` back to TiXI, then reopen a fresh TiGL handle (required —
  the C-API caches `wingGetReferenceArea` on a handle and `Invalidate` does not
  bust it) and swap it into the session. Returns the MEASURED geometry read back
  from the fresh handle. Regression test: `tests/test_morph_wing_coupling.py`
  (asserts `get_wing_summary` reports the new span/area/AR, the achieved values
  match the targets, and the exported BREP bytes change). `target_area_m2` /
  `target_aspect_ratio` use the FULL-planform convention (the tool doubles the
  reference area for mirror-symmetric wings), so aviary's AREA / ASPECT_RATIO
  design variables can be passed directly — e.g. area 130.1 / AR 15.6 yields
  span 45 m, matching the F25 nominal.
- New `morph_fuselage` tool — same idea for the fuselage: scales length (x) and
  diameter (y,z) via the fuselage transformation, rebuilds, and reports the new
  dimensions. Current/measured dimensions come from the exported BREP's
  OpenCASCADE bounding box, deliberately avoiding the native fuselage geometry
  queries (`fuselageGetVolume`/`fuselageGetLength`), which segfault on some CPACS
  files (e.g. the D150) and crash the server uncatchably.
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
