# IDEA

Explore ideas through evolution. Generate a population, compare candidates,
combine their ingredients, and follow what changes across generations.

**[Open IDEA](https://idea-evolution.cloud.run/)** — sign in with Google and add
your Gemini or OpenAI API key in **Settings**.

Choose a template, model, population and budget, then start. Open any idea to
read it, inspect its evaluations and genes, or trace its parents. The gene
history shows shared ingredients across the whole run; selecting an idea
highlights its DNA.

Built-in templates cover **100-word stories, game design, arcade games and AI
research proposals**. Create or customize templates in the app.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
[Firebase CLI](https://firebase.google.com/docs/cli) and Java JDK 21+.

```bash
git clone https://github.com/rbrthogan/idea.git
cd idea
uv sync --locked
./run_local.sh
```

Open **[localhost:8000](http://127.0.0.1:8000/)**. The local launcher uses the
Firestore emulator and development sign-in. Add a provider key in Settings.
Stop with **Ctrl+C** and let the emulator finish saving before closing the
terminal.

Model calls cost money. Start with a small population and budget. Displayed
costs are estimates from reported usage; running or failed requests can still
incur charges. Provider billing is the final authority.

## Development

```bash
uv run pytest -q --strict-markers
```

The default suite is offline and does not make paid model calls.

- [Local setup and troubleshooting](docs/LocalDevelopment.md)
- [How evolution works](docs/EvolutionOverview.md)
- [Template authoring](idea/prompts/templates/README.md)
- [Deploy and roll back](DEPLOY.md)

IDEA uses FastAPI, plain JavaScript, Firebase Authentication and Firestore.
Local evolutions run in a supervised task; production uses Cloud Run Jobs.
Generated-code execution is archived; this release evolves text ideas.
