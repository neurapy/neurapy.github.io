# PINNfluence ICML Demo

## Install

The webdemo requires Node.js 24. The repo includes `.nvmrc`; with `nvm`, run:

```bash
nvm install
nvm use
```

If `nvm` is not installed, install it first:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.5/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install
nvm use
```

```bash
make install # install dependencies
```

## Deploy from Local machine

```bash
make build    # Build deployable Webdemo in webdemo/dist
make preview  # Start a Service on 0.0.0.0.
```

Reachable from local Network. For non-local, set up port forwarding.

## Development-Preview

```bash
make dev # Hot-Reloading, no build
```

## Github.io hosting

Currently deployed at `neurapy.github.io`.

Pushed Changes automatically update deployment.

On Every push Github:

1. installs the frontend dependencies from `webdemo/package-lock.json`,
2. builds the static app with `npm --prefix webdemo run build`,
3. uploads only `webdemo/dist/` to GitHub Pages.

Manual Deployment

```bash
make deploy
```

Check the deployment:

```bash
make deploy-status
make deploy-watch
```

Print the command to disable GitHub Pages:

```bash
make undeploy
```

#### Deploy in different Repo

1. Have `gh` installed and authenticated.
2. Create new Repo
3. Add upstream

```bash
git remote add newrepo git@github.com:OWNER/REPO.git
git push newrepo main
```
1. Enable pages

```bash
gh api --method POST repos/OWNER/REPO/pages -f build_type=workflow # If Pages are set up but configured differently for the repo use PUSH instead of POST
```
1. Change `PAGES_REPO` in the `Makefile`
2. `Make deploy`

(Or just do it manually in the Web. Didn't work for me, however.)

## Generate Data from raw_data

If you have `raw_data/` but `webdemo/public/data` is missing or stale, regenerate the static Webdemo bundle with the canonical script:

```bash
src/download_model_zoo_data.sh --build-only
make verify-data
```

The script writes schema-v9 static data. Each problem must have paired `*_good` and `*_bad` raw-data folders, for example `burgers_float64_good` and `burgers_float64_bad`. Influence matrices are exported as dense `scores.f32` files for HTTP byte-range loading.

## No Raw Data?

Expected in `raw_data/` are paired folders like `allen_cahn_float64_good` and `allen_cahn_float64_bad`, which have to contain:

```path
..._influence_scores/
..._validation/
..._full.pt
```

`src/download_model_zoo_data.sh --build-static-demo-data` can download the configured artifacts from `ai-ws-213` and then regenerate the Webdemo bundle. Edit the folder/prefix blocks in that script before using it against a different source.
