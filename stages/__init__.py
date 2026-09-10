"""KFP pipeline stages, handing off through a file.

Two stages, each an independently executable step:

    quality_screen           -> quality_screen.jsonl          (GPU)
    evaluate_quality_screen  -> quality_screen_report.json    (CPU)

They are separate because they want opposite hardware: the manifest gives the
screen every available GPU and the scorer none. The handoff is a file rather
than a call for the same reason -- the two never share a process.

There is no `clean` stage between them. `clean` normalised free-text field
values before comparison; the screen's answers are fixed tokens, so there is
nothing to normalise.
"""
