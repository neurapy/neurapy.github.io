# Static PINNfluence Demo

Serve the repository root and open `/webdemo/`:

```bash
python3 -m http.server 8080
```

The app loads generated assets from `webdemo/data/`. Build those assets with:

```bash
.venv/bin/python src/build_static_demo_data.py \
  --problems allen_cahn_float64 \
  --matrix-mode core \
  --k-max 200 \
  --overwrite
```


```chagpt

• Use this from the repo root:

  .venv/bin/python src/build_static_demo_data.py \
    --matrix-mode core \
    --k-max 200 \
    --field-points 60000 \
    --overwrite \
    --skip-incomplete

  This processes all data/*_float64 folders, uses dense precomputed display fields, keeps the web
  bundle practical, and includes the four main matrices per run:

  - PINNfluence / total loss -> output 0
  - PINNfluence / total loss -> total loss
  - GradDot / total loss -> output 0
  - GradDot / total loss -> total loss

  Then verify:

  .venv/bin/python src/verify_static_demo_data.py --samples 5

  Start/open the app:

  python3 -m http.server 8080

  Then go to:

  http://127.0.0.1:8080/webdemo/

  I would not use --matrix-mode all for the public demo unless you really need every BC/PDE
  cross-term; it will make the static bundle much larger. - THat is bullshit. Use it.
```
