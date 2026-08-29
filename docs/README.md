# ABLE project page

This directory contains the dependency-free static website for ABLE. GitHub
Pages can serve it directly; no build step or deployment workflow is required.

## Preview locally

From the repository root, run:

```bash
python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>.

## Publish with GitHub Pages

After pushing the repository to GitHub:

1. Open the repository's **Settings → Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**.
3. Select the `main` branch and the `/docs` folder, then click **Save**.
4. Wait for GitHub Pages to finish the first deployment.

The project page will be available at
<https://ziiroo1126.github.io/ABLE/>. Later changes under `docs/` are published
automatically after they are pushed to `main`.

The `.nojekyll` marker asks GitHub Pages to serve these static files without
Jekyll processing.
