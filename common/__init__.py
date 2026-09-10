"""Shared building blocks for the image-quality screen.

Intentionally a marker with no re-exports. It used to pull five names up from
`pipeline_config` so callers could write `from common import PipelineConfig`;
nothing does any more, and a re-export means a name has two import paths that
can drift.

Kept rather than deleted for two reasons. `stages/`, `models/` and
`models/backends/` all have one, and a package that is explicit in three places
and implicit in the fourth invites the question of which is deliberate. More
concretely: without this file `common` becomes an implicit namespace package,
and any other directory called `common` on `sys.path` would silently merge into
it rather than conflict.
"""
