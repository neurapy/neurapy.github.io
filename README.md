# PINNfluence ICML Demo

## Install

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
<<<<<<< HEAD
make verify-data
=======
make dev # Hot-Reloading, no build
>>>>>>> 18bed37 (Refresh Makefile webdemo targets)
```

## Github.io hosting

<<<<<<< HEAD
```bash
make webdemo-build
```
=======
Currently deployed at `neurapy.github.io`.
>>>>>>> 18bed37 (Refresh Makefile webdemo targets)

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
<<<<<<< HEAD
make deploy-status
make deploy-watch
=======
make undeploy
>>>>>>> 18bed37 (Refresh Makefile webdemo targets)
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
<<<<<<< HEAD
make deploy
=======
uv run python src/build_static_demo_data.py \
--matrix-mode core \
--max_local_influence_points 64 \
--raster-max-resolution 1024 \
--overwrite \
--workers 8 \

# matrix-mode all generates more plots
# max_local_influence_points is  max number of influence point per candidate_point
# raster-max-resolution is resolution of precomputed prediction / loss graphs on the longest axis

make verify-data # Verification step
>>>>>>> 18bed37 (Refresh Makefile webdemo targets)
```

## No Raw Data?

Expected in `($"PROJECT_ROOT")/raw_data/` are your folders like `allen_cahn_float64`, which have to contain:
```path
..._influence_scores/
..._validation/
..._full.pt
```

There is also a script to automatically download those from my folder from the `ai-ws-213`. Comment out / fill in what you need. Its faster with an SSH-Key.

Have fun on ICML


---

# Notes

## Webapp ToDos

### Implement
1. Aspect Ration richtig nutzen um Prediction + Global Influence am best möglichsten darszustellen
- Die Bilder müssen richtig scalen!
- Das was Aleks geschrieben hat
- Region Influence
- Mobile version
- Sinnvolle Defaults ausgewählt

### Performance Optims

1. Erst die Sachen laden, die man sieht. 
2. Point wise Influence Matrix Lazy im Hintergrund laden. Immer, wenn wir einen einzelnen Batch haben wollen und die jeweilige Problemmatrix
noch nicht haben, dann fetchen wir nur diesen Batch.

### Webdemo

Der nächste Schritt ist, dass ich mir genau überlege, was ich alles brauche. Ich habe bereits eine gute Base. 
Es sollte eine .js Webdemo werden. Das gibt mir die meisten Freiheiten.
Der hintergrund sollte precomputed sein. Darüber sollten die Trainingspunkte markiert sein. Ich sollte darauf klicken können und in einer anderen Grafik ändern sich die Influences. Slider für Top K. Etc. 

Aleks:

```
* ziel: github.io seite fürs paper 
    * mit links zu arxiv, code, demo
    * oder demo direkt drauf
* Auf jeden Fall dabei:
    * Point selection
    * Und Visualisierung von top k influential points 
        * Alternativ Visualisierung aller Influences
    * Visualisierung der Influence des gegebenen Punktes
* Figure 4: auch region selection wäre supi
* Figure 5: loss decomposition sollte drinnen sein

```
