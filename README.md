# PINNfluence ICML Demo

## How to run

Serve only the static webdemo:

```bash
make install
make webdemo
```
Alternatively and only if this fails: `python3 -m http.server 8080` from webdemo

Then open <http://127.0.0.1:8080/>.

The rest should be not necessary. But just in case something goes wrong:

## Generate Data from raw_data


The app serves pre-generated static assets from `webdemo/data/`.

Build or refresh the demo bundle from the repo root:

```bash
uv run python src/build_static_demo_data.py --matrix-mode core --k-max 150 --raster-max-resolution 1024 --overwrite --matrix-workers 8
```

Verify the generated assets:

```bash
uv run python src/verify_static_demo_data.py --samples 5
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
---
---

# Notes

## Webapp ToDos

### Implement
- Die Bilder müssen richtig Scalen!
- Das was Aleks geschrieben hat
- Region Influence
- Mobile version
- SInnvolel Defaults ausgewählt

### Performance Optims

1. Erst die Sachen laden, die man sieht. 
2. Point wise Influence Matrix Lazy im Hintergrund laden. Immer, wenn wir einen einzelnen Batch haben wollen und die jeweilige Problemmatrix
noch nicht haben, dann fetchen wir nur diesen Batch.


## Was muss ich machen? 

1. Richtige Arrays runter laden und in data/
2. Visualisieren um zu schauen, dass alles stimmt. Lokale Demo schreiben.
3. Richtiges Tool für den Job finden. Vlt:
  - Py library
  - Javascript Only
  - Python Backend + Javascript Frontend


### 1. Daten:
Was brauche ich?

Probleme:
- Allen-Cahn | allen_cahn_float64
``
- Burgers | burgers_float64
- drift-diffusion | drift_diffusion_float64
- wave | wave_float64

> Erstmal diese 4. Heat finde ich nicht aufm server

- heat? 

> Zusätzlich dann noch sowas wie Navier stokes etc. 

Ich brauche train_x und candidate_point key in den files
#### Allen-Cahn

`allen_cahn_adam_100000_adam_25000_lbfgs_2500_domain_500_boundary_500_initial_3_x_64_hidden_float64_True_12_soft_influence_scores`
`allen_cahn_adam_100000_adam_25000_lbfgs_2500_domain_500_boundary_500_initial_3_x_64_hidden_float64_True_12_soft_validation`
`allen_cahn_adam_100000_adam_25000_lbfgs_2500_domain_500_boundary_500_initial_3_x_64_hidden_float64_True_12_soft_full.pt`
-> Daraus kann ich andere Arrays generieren, die ich dann benutze, ähnlich, wie Aleks es gemacht hatte.

> DONE

---

### Webdemo

Der nächste Schritt ist, dass ich mir genau überlege, was ich alles brauche. Ich habe bereits eine gute Base. 
Es sollte eine JS Webdemo werden. Das gibt mir die meisten Freiheiten.
Der hintergrund sollte precomputed sein. Darüber sollten die Trainingspunkte markiert sein. Ich sollte darauf klicken können und in einer anderen Grafik ändern sich die Influences. SLider für Top K. Etc. 


### Aleks Hinweise

```
bin noch dabei die repo aufzuräumen aber aknn dir heute schonmal ne "dreckige" version geben
wo die punkte generiert werden
[1:52 PM]glaube aber die habe ich jeweils immer mit abgespeichert
[1:53 PM]schau mal ob du nen train_x und candidate_point key in den files findest
```

```
[4:23 PM]in den pt checkpoints hast du nen key mit train_x_all
[4:23 PM]das sind die train points
[4:23 PM]dann in den influence ordnern hast du jeweils bei jedem npz file ein 'candidate_points' key
[4:23 PM]das sind die test punkte auf denen es ausgewertet wurde
[4:24 PM](sind nicht die gleichen wie test_x glaube)
```

```
[6:10 PM]valider key:

allen_cahn_adam_100000_adam_25000_lbfgs_2500_domain_500_boundary_500_initial_3_x_64_hidden_float64_True_11_soft_full.pt
valide runs mit influences haben gleicher name (ohne _full am ende) _influence_scores  suffix 
da drinnen sind: 
influences_von_auf.npz 

train_x_all  ist im .pt  checkpoint als key 
candidate_points (test punkte) sind in den jeweiligen influence files als key 
[6:14 PM]unter notebooks/_ICML_REBUTTAL_CHECK_NNCG/problem_name.ipynb  müsstest du immer ein example finden wie man das lädt
Aleks Krasowski  [6:21 PM]
steamlit.io ist ultra easy, sonst einfach claude fragen was es da für nicen stuff gibt lol
```

```

from .. import data

DEFAULTS = {
    "lr": 0.001,
    "layers": [2] + [32] * 3 + [1],
    "n_iterations": 0,
    "n_iterations_lbfgs": 0,
    "num_domain": 1000,
    "k": 2,
    "c": 0,
    "optimizer": "adam",
    "device": "cpu",
    "seed": 42,
    "train_distribution": "Hammersley",
    "model_zoo_src": "./model_zoo",
    "model_zoo": "./model_zoo",
    "DATASET_DIR": data.__path__[0],
    "base_optimizer": "adam",
}

PROBLEMS = {
    "allen_cahn": {
        "layers": [2, 64, 64, 64, 1],
        "num_domain": 2500,
        "num_boundary": 500,
        "num_initial": 500,
        "n_iterations": 100_000,
        "n_iterations_lbfgs": 25_000,
        "seeds": [0, 1, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    "burgers": {
        "layers": [2, 32, 32, 32, 1],
        "num_domain": 2500,
        "num_boundary": 500,
        "num_initial": 500,
        "n_iterations": 50_000,
        "n_iterations_lbfgs": 12_000,
        "seeds": list(range(10)),
    },
    "diffusion": {
        "layers": [2, 32, 32, 32, 1],
        "num_domain": 1000,
        "num_boundary": 100,
        "num_initial": 100,
        "n_iterations": 15000,
        "n_iterations_lbfgs": 5000,
        "seeds": list(range(10)),
    },
    "drift_diffusion": {
        "layers": [2, 64, 64, 64, 1],
        "num_domain": 1000,
        "num_boundary": 100,
        "num_initial": 100,
        "n_iterations": 15_000,
        "n_iterations_lbfgs": 5_000,
        "seeds": list(range(10)),
    },
    "wave": {
        "layers": [2, 100, 100, 100, 100, 100, 1],
        "num_domain": 2500,
        "num_boundary": 500,
        "num_initial": 500,
        "n_iterations": 100_000,
        "n_iterations_lbfgs": 25_000,
        "seeds": list(range(10)),
    },
    "navier_stokes_nd": {
        "layers": [2, 64, 64, 64, 3],
        "num_domain": 7500,
        "num_boundary": 2500,
        "num_initial": 0,
        "n_iterations": 100_000,
        "n_iterations_lbfgs": 25_000,
        "seeds": list(range(10)),
    },
    "poisson_disk": {
        "layers": [2, 32, 32, 32, 1],
        "num_domain": 2500,
        "num_boundary": 500,
        "num_initial": 0,
        "n_iterations": 50_000,
        "n_iterations_lbfgs": 12_000,
        "seeds": list(range(10)),
    },
}

BAD_PROBLEMS = {
    "allen_cahn": {
        "layers": [2, 64, 64, 64, 1],
        "num_domain": 2500,
        "num_boundary": 500,
        "num_initial": 500,
        "n_iterations": 100_000,
        "n_iterations_lbfgs": 0,
        "seeds": list(range(10)),
    },
    "burgers": {
        "layers": [2, 32, 32, 32, 1],
        "num_domain": 500,
        "num_boundary": 100,
        "num_initial": 100,
        "n_iterations": 50_000,
        "n_iterations_lbfgs": 12_000,
        "seeds": list(range(10)),
    },
    "diffusion": {
        "layers": [2, 32, 32, 32, 1],
        "num_domain": 10,
        "num_boundary": 2,
        "num_initial": 2,
        "n_iterations": 15000,
        "n_iterations_lbfgs": 5000,
        "seeds": list(range(10)),
    },
    "drift_diffusion": {
        "layers": [2, 64, 64, 64, 1],
        "num_domain": 200,
        "num_boundary": 20,
        "num_initial": 20,
        "n_iterations": 15_000,
        "n_iterations_lbfgs": 5_000,
        "seeds": list(range(10)),
    },
    "navier_stokes_nd": {
        "layers": [2, 64, 64, 64, 3],
        "num_domain": 1500,
        "num_boundary": 500,
        "num_initial": 0,
        "n_iterations": 100_000,
        "n_iterations_lbfgs": 25_000,
        "seeds": list(range(10)),
    },
    "wave": {
        "layers": [2, 100, 100, 100, 100, 100, 1],
        "num_domain": 2500,
        "num_boundary": 500,
        "num_initial": 500,
        "n_iterations": 100_000,
        "n_iterations_lbfgs": 0,
        "seeds": list(range(10)),
    },
    "poisson_disk": {
        "layers": [2, 32, 32, 32, 1],
        "num_domain": 100,
        "num_boundary": 20,
        "num_initial": 0,
        "n_iterations": 15_000,
        "n_iterations_lbfgs": 5_000,
        "seeds": list(range(10)),
    },
}
```
