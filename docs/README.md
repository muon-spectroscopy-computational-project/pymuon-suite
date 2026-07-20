How to work with the documentation:

* install the docs dependencies with `uv sync --group docs` (run from the repository root)
* launch the local development server from this `docs/` directory with `uv run mkdocs serve`
* the site is built and deployed to GitHub Pages automatically by the
  `.github/workflows/docs.yml` workflow whenever changes to `docs/**` are pushed to `main`;
  it can also be triggered manually from the Actions tab (`workflow_dispatch`)

The pages themselves live in `docs/docs/*.md` and the navigation is defined in
`docs/mkdocs.yml`.

Warning: when adding or editing math equations, make sure to have a blank line after the
closing `$$` tag. For some reason, without it, MathJax rendering can fail.
