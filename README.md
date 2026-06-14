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
4. Enable pages
```bash
gh api --method POST repos/OWNER/REPO/pages -f build_type=workflow # If Pages are set up but configured differently for the repo use PUSH instead of POST
```
5. Change `PAGES_REPO` in the `Makefile`
6. `Make deploy`

(Or just do it manually in the Web. Didn't work for me, however.)

## Generate Data from raw_data

If you have `raw_data/` but `webdemo/public/data` is missing: 

```bash
rm -rf webdemo/public/data
uv run python src/build_static_demo_data.py \
--matrix-mode core \
--max_local_influence_points 1000 \
--n-candidate 1000 \
--n-train 1000 \
--raster-max-resolution 1024 \
--field-batch-size 8192 \
--overwrite \
--workers 8 \

# matrix-mode all generates more plots
# matrix-mode core exports the main PINNfluence output/loss matrices:
# influences_total_loss_output_0, influences_total_loss_output_1,
# influences_total_loss_output_2, and influences_total_loss_total_loss
# each influence matrix is exported as one dense scores.f32 file for HTTP byte-range loading
# max_local_influence_points is max number of influence points shown per candidate point
# n-candidate / n-train control how many existing raw points are exported
# raster-max-resolution is resolution of precomputed prediction / loss graphs on the longest axis
# field-batch-size controls prediction / loss raster inference batches; lower it if CPU RAM is tight
# static demo data uses schema v8: every problem must have exactly one *_good
# and one *_bad raw-data folder, for example burgers_float64_good and
# burgers_float64_bad. Legacy unsuffixed folders are ignored.

make verify-data # Verification step
```

## No Raw Data?

Expected in `($"PROJECT_ROOT")/raw_data/` are paired folders like `allen_cahn_float64_good` and `allen_cahn_float64_bad`, which have to contain:
```path
..._influence_scores/
..._validation/
..._full.pt
```

There is also a script to automatically download those from my folder from the `ai-ws-213`. Comment out / fill in what you need. Its faster with an SSH-Key.

Have fun on ICML


---

# Notes

## Schreiben an Aleks
- Schick mir bitte die Folder names mit GOOD and BAD model. Und falls es gibt, zu diesen Modellen auch aufgesplittet nach den IC Gesplitteten Graphs. (Falls es sie nicht gibt, generiere ich sie selbst.)

## TODOS nach ALeks quatschen
- Aleks schickt mir Folder names für GOOD und BAD model. 
-> Neue Daten Generieren
-> Good / Bad switcher in der TOP BAR
- Kontur in Train rein
- Evtl. Den Nach Loss Terms gesplittete Global Influence POlot
- Impressum & Datenschutzerklärung rein (Aleks schätzt die IT schickt nochma was)
- Drift Diffusion 1:1
- Selfhosting configuren? Warte auf was die it zu aleks sagt zu wie wir hosten
- Loss nicht mehr nach unterschiedlichen Terms aufsplitten
- Checkbox, und ansonsten nichts preloaden.
- Axis Labels und Colorbar
- When you click outside the Graph, select a Boundary Point!!
- Extra selector to not increase Influence points in size.



- Manchmal clustern Influences an bestimmten regionen von boundaries. Wieso?
