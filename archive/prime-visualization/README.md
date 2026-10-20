# Archived visualization prototype

Generated-code evolution is retired from the released web app and worker.
New, resumed and continued artifact runs are rejected, including requests from
older clients. Environment flags cannot re-enable execution. Historical
results, previews and verified downloads remain readable.

The original [template](idea/prompts/templates/prime_visualization.yaml) is kept
here for compatibility tests and reference. It is not in the active template
picker or deployment image. The earlier implementation is available in Git
history; duplicated UI and implementation snapshots are not release inputs.
Generic evaluator and artifact-reader contracts remain in the application.

Any restoration should happen on a separate branch, with explicit work on
execution isolation, bounded costs, durable checkpoints and artifacts,
cancellation and failure recovery. Copying the template into the active folder
or setting an old feature flag is not sufficient.
