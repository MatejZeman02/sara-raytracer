# CTU FIT NI-PG1 Matej Zeman (zemanm40) 2026

My Sara raytracer in numba-cuda. Semestral work in PG1 taught by Ing. Radek Richtr Ph.D on CTU FIT in Prague.

## Structure/Installation

Separate homeworks are in the branches (hw0x). The main source code is inside `src/` folder. It is using my `utils/` module.

For Python package on local machine, I use conda venv called 'raytracer'. Use `run_local.sh` script to run specific branch.

I managed to run it on FIT cluster as well with using standard `.venv/` installed from `requirements.txt`
If you do not have conda package manager, build your `.venv` form `requirements.txt`.
OIDN is optional; without it the renderer still runs and simply skips the denoise pass.


The output will be in the `src/output/` folder as `output.ppm/png/jpg`.

### TinyObjLoader

I use the c++ header tiny-obj-loader library together with python bindings. I parsed the triangles there too. So user on different machine needs to recompile it for there is older version of python on the cluster or your device.

Use `./build_tinyobjloader.sh.sh python3.<version>` for compilation from the root directory. Note: For each homework branch may be different version of the library. So recompiling when switching branches is adviced. You can specify your version of python to compile for, but default is `python3.14`.


## Homework renders

<img src="src/output/hw01/output.png" width="25%"/>
<img src="src/output/hw02/output.jpg" width="25%"/>

***

<img src="src/output/hw03/output.jpg" width="25%"/>
<img src="src/output/hw04/output.jpg" width="25%"/>

***

<img src="src/output/hw05/output.jpg" width="25%"/>


## Render Times

Testing on my box-advanced with +-5500 triangles.
The meassurements assume the data is already on gpu.

> Dimensions per machine:
> - 1440**2 on local rtx3070 mobile.
> - 5760**2 on remote A100 on cluster.
> - The scenes are ussualy 5 units (blender meters) tall.


## Performance Log:
Saved bash rendering on cpu/gpu with/without BVH:

```bash
GPU_DIMENSION = 1024
SCENE_NAME = "bunny"
SAMPLES = 6
DENOISE = False
MAX_BOUNCES = 16

Runs on device: GPU
[timing] init python         :    2.65 s
[timing] bvh build           :    9.05 s
[timing] init cuda + alloc   :    0.27 s
[timing] jit compile run     :    6.33 s
[timing] render (no ds)      :   12.11 s

=================================================================
  STATISTICS (No DS on GPU)
=================================================================
Resolution:             1024 x 1024 (1,048,576 pixels)
Render time:            12.109 s
Throughput (whole run): 0.29 MRays/s
-----------------------------------------------------------------
RAY DISTRIBUTION (Total: 3,488,607)
  Primary:               30.1%  (1,048,576)
  Secondary:             32.7%  (1,141,702)
  Shadow:                37.2%  (1,298,329)
-----------------------------------------------------------------
WORKLOAD DISTRIBUTION
  Sky (1 ray):           24.2%  (254,098)
  Standard geometry:     73.3%  (768,384)
  Hard (>> avg tests):    2.5%  (26,094)
-----------------------------------------------------------------
BVH EFFICIENCY (Total incidence ops: 236,004,172,848)
  Node/Triangle ratio:  0.0 : 1
  Avg ops per ray:      67650.0
  Avg nodes per ray:    0.0 (Ideal O(logN) ~ 16.1)
-----------------------------------------------------------------

PER_HIT-PIXEL LOAD (min / mean / max)
  Rays calls:        2 / 4.1 / 14
  Incidence tests:   69655 / 274839.3 / 972482
=================================================================

[timing] render (with ds)    :   12.08 s (0.1 FPS)
[timing] copy hdr to host    :    0.01 s
[timing] postprocess (srgb/tonemapper on CPU):    0.68 s

=================================================================
  STATISTICS (DS on GPU)
=================================================================
Resolution:             1024 x 1024 (1,048,576 pixels)
Render time:            12.766 s
Throughput (whole run): 0.27 MRays/s
-----------------------------------------------------------------
RAY DISTRIBUTION (Total: 3,488,542)
  Primary:               30.1%  (1,048,576)
  Secondary:             32.7%  (1,141,794)
  Shadow:                37.2%  (1,298,172)
-----------------------------------------------------------------
WORKLOAD DISTRIBUTION
  Sky (1 ray):           24.2%  (254,084)
  Standard geometry:     73.3%  (768,399)
  Hard (>> avg tests):    2.5%  (26,093)
-----------------------------------------------------------------
BVH EFFICIENCY (Total incidence ops: 235,996,564,239)
  Node/Triangle ratio:  0.0 : 1
  Avg ops per ray:      67649.1
  Avg nodes per ray:    0.0 (Ideal O(logN) ~ 16.1)
-----------------------------------------------------------------

PER_HIT-PIXEL LOAD (min / mean / max)
  Rays calls:        2 / 4.1 / 14
  Incidence tests:   69613 / 274826.1 / 972482
=================================================================
Click to see the result onto: /home/zemanm40/ni-gpu/zemanm40/src/output/output.jpg

[timing] total (+-)          :   43.71 s
```

```bash
GPU_DIMENSION = 1024
SCENE_NAME = "box-spheres"
SAMPLES = 6
DENOISE = False
MAX_BOUNCES = 16

Runs on device: GPU
[timing] init python         :    2.72 s
[timing] bvh build           :    6.84 s
[timing] init cuda + alloc   :    0.27 s
[timing] jit compile run     :    5.14 s
[timing] render (no ds)      :    0.31 s

=================================================================
  STATISTICS (No DS on GPU)
=================================================================
Resolution:             1024 x 1024 (1,048,576 pixels)
Render time:            0.306 s
Throughput (whole run): 6.85 MRays/s
-----------------------------------------------------------------
RAY DISTRIBUTION (Total: 2,091,569)
  Primary:               50.1%  (1,048,576)
  Secondary:              8.1%  (168,681)
  Shadow:                41.8%  (874,312)
-----------------------------------------------------------------
WORKLOAD DISTRIBUTION
  Sky (1 ray):           25.0%  (261,660)
  Standard geometry:     69.6%  (729,659)
  Hard (>> avg tests):    5.5%  (57,257)
-----------------------------------------------------------------
BVH EFFICIENCY (Total incidence ops: 4,378,508,862)
  Node/Triangle ratio:  0.0 : 1
  Avg ops per ray:      2093.4
  Avg nodes per ray:    0.0 (Ideal O(logN) ~ 11.1)
-----------------------------------------------------------------

PER_HIT-PIXEL LOAD (min / mean / max)
  Rays calls:        2 / 2.3 / 32
  Incidence tests:   2194 / 4836.6 / 66455
=================================================================

[timing] render (with ds)    :    0.22 s (4.6 FPS)
[timing] copy hdr to host    :    0.01 s
[timing] postprocess (srgb/tonemapper on CPU):    0.57 s

=================================================================
  STATISTICS (DS on GPU)
=================================================================
Resolution:             1024 x 1024 (1,048,576 pixels)
Render time:            0.799 s
Throughput (whole run): 2.62 MRays/s
-----------------------------------------------------------------
RAY DISTRIBUTION (Total: 2,092,266)
  Primary:               50.1%  (1,048,576)
  Secondary:              8.1%  (168,963)
  Shadow:                41.8%  (874,727)
-----------------------------------------------------------------
WORKLOAD DISTRIBUTION
  Sky (1 ray):           25.0%  (261,667)
  Standard geometry:     69.6%  (729,652)
  Hard (>> avg tests):    5.5%  (57,257)
-----------------------------------------------------------------
BVH EFFICIENCY (Total incidence ops: 4,379,806,720)
  Node/Triangle ratio:  0.0 : 1
  Avg ops per ray:      2093.3
  Avg nodes per ray:    0.0 (Ideal O(logN) ~ 11.1)
-----------------------------------------------------------------

PER_HIT-PIXEL LOAD (min / mean / max)
  Rays calls:        2 / 2.3 / 32
  Incidence tests:   2194 / 4838.3 / 70016
=================================================================
Click to see the result onto: /home/zemanm40/ni-gpu/zemanm40/src/output/output.jpg

[timing] total (+-)          :   16.36 s
```

***

```bash

GPU_DIMENSION = 1024
SCENE_NAME = "bunny"
SAMPLES = 6
DENOISE = True
MAX_BOUNCES = 16

Runs on device: GPU
[timing] init python         :    1.32 s
[timing] init cuda + alloc   :    1.90 s
[timing] jit compile run     :    2.47 s
[timing] render (no ds)      :   39.62 s

=================================================================
  STATISTICS (No DS on GPU)
=================================================================
Resolution:             1024 x 1024 (1,048,576 pixels)
Render time:            39.622 s
Throughput (whole run): 0.09 MRays/s
-----------------------------------------------------------------
RAY DISTRIBUTION (Total: 3,487,462)
  Primary:               30.1%  (1,048,576)
  Secondary:             32.7%  (1,141,210)
  Shadow:                37.2%  (1,297,676)
-----------------------------------------------------------------
WORKLOAD DISTRIBUTION
  Sky (1 ray):           24.2%  (254,094)
  Standard geometry:     73.3%  (768,577)
  Hard (>> avg tests):    2.5%  (25,905)
-----------------------------------------------------------------
BVH EFFICIENCY (Total incidence ops: 235,929,968,622)
  Node/Triangle ratio:  0.0 : 1
  Avg ops per ray:      67650.9
  Avg nodes per ray:    0.0 (Ideal O(logN) ~ 16.1)
-----------------------------------------------------------------

PER_HIT-PIXEL LOAD (min / mean / max)
  Rays calls:        2 / 4.1 / 14
  Incidence tests:   69656 / 274744.8 / 972482
=================================================================

[timing] render (with ds)    :   39.62 s (0.0 FPS)
[timing] copy hdr to host    :    0.00 s
[timing] oidn denoise        :    0.93 s
[timing] postprocess (srgb/tonemapper on CPU):    0.96 s

=================================================================
  STATISTICS (DS on GPU)
=================================================================
Resolution:             1024 x 1024 (1,048,576 pixels)
Render time:            41.515 s
Throughput (whole run): 0.08 MRays/s
-----------------------------------------------------------------
RAY DISTRIBUTION (Total: 3,488,533)
  Primary:               30.1%  (1,048,576)
  Secondary:             32.7%  (1,141,668)
  Shadow:                37.2%  (1,298,289)
-----------------------------------------------------------------
WORKLOAD DISTRIBUTION
  Sky (1 ray):           24.2%  (254,098)
  Standard geometry:     73.3%  (768,320)
  Hard (>> avg tests):    2.5%  (26,158)
-----------------------------------------------------------------
BVH EFFICIENCY (Total incidence ops: 235,995,340,763)
  Node/Triangle ratio:  0.0 : 1
  Avg ops per ray:      67648.9
  Avg nodes per ray:    0.0 (Ideal O(logN) ~ 16.1)
-----------------------------------------------------------------

PER_HIT-PIXEL LOAD (min / mean / max)
  Rays calls:        2 / 4.1 / 14
  Incidence tests:   69655 / 274828.2 / 972482
=================================================================
[timing] total (+-)          :   86.95 s
```


<!-- ***
```bash
=================================================================
  STATISTICS (No DS on CPU)
=================================================================
Resolution:             1440 x 1440 (2,073,600 pixels)
Render time:            111.293 s
Throughput (whole run): 0.14 MRays/s
-----------------------------------------------------------------
RAY DISTRIBUTION (Total: 15,256,468)
  Primary:               13.6%  (2,073,600)
  Secondary:             41.5%  (6,330,831)
  Shadow:                44.9%  (6,852,037)
-----------------------------------------------------------------
BVH EFFICIENCY (Total incidence ops: 80,393,952,357)
  Node/Triangle ratio:  0.0 : 1
  Avg ops per ray:      5269.5
  Avg nodes per ray:    0.0 (Ideal O(logN) ~ 12.4)
-----------------------------------------------------------------
PER-PIXEL LOAD (min / mean / max)
  Rays calls:        1 / 7.4 / 32
  Incidence tests:   5492 / 38770.2 / 175744
=================================================================
```
***
```bash
=================================================================
  STATISTICS (DS on CPU)
=================================================================
Resolution:             1440 x 1440 (2,073,600 pixels)
Render time:            1.091 s
Throughput (whole run): 13.98 MRays/s
-----------------------------------------------------------------
RAY DISTRIBUTION (Total: 15,256,462)
  Primary:               13.6%  (2,073,600)
  Secondary:             41.5%  (6,330,826)
  Shadow:                44.9%  (6,852,036)
-----------------------------------------------------------------
BVH EFFICIENCY (Total incidence ops: 608,288,602)
  Node/Triangle ratio:  7.1 : 1
  Avg ops per ray:      39.9
  Avg nodes per ray:    34.9 (Ideal O(logN) ~ 12.4)
-----------------------------------------------------------------
PER-PIXEL LOAD (min / mean / max)
  Rays calls:        1 / 7.4 / 32
  Incidence tests:   1 / 293.3 / 5373
=================================================================
``` -->


## Launching debug in VS Code

Use this `.vscode/launch.json` setup:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "debug raytracer",
            "type": "debugpy",
            "request": "launch",
            "module": "src.main",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                // added workspaceRoot/src, for absolute imports
                "PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/src"
            }
        }
    ]
}
```



***

## Differences with C++ CUDA (in czech from chat)

Shrnutí převodu konceptů z CUDA C++ (`nvcc`) do Numby:

### 1. Kompilační flagy a optimalizace (nvcc -> Numba)

- **`-O3` (Optimalizace CPU/GPU)**
- **V Numbě:** *Není potřeba.* Numba využívá LLVM a maximální optimalizace aplikuje automaticky.


- **`-lineinfo` a `-G` (Debug)**
- **V Numbě:** `@cuda.jit(lineinfo=True)` nebo `@cuda.jit(debug=True)`
- **Funkcionalita:** `lineinfo` propojuje PTX kód se zdrojovým kódem Pythonu pro Nsight Compute (bez ztráty výkonu). `debug` navíc přidává asserty a kontroly mezí polí (výrazně zpomaluje běh).


- **`-maxrregcount=X` (Limit registrů)**
- **V Numbě:** `@cuda.jit(max_registers=X)`
- **Funkcionalita:** Zabrání překladači použít příliš mnoho registrů na vlákno, což může pomoci spustit více vláken najednou (zvýšit occupancy).


- **`-arch` a `-code` (Cílová architektura)**
- **V Numbě:** *Není potřeba.* Numba je JIT (Just-In-Time) kompilátor. Architekturu (např. `sm_86`) si detekuje dynamicky podle karty, na které kód zrovna běží.


- **`-use_fast_math`, `-ftz`, `-prec-div`, `-prec-sqrt` (Rychlá matematika)**
- **V Numbě:** `@cuda.jit(fastmath=True)`
- **Funkcionalita:** Povoluje HW optimalizace, zaokrouhluje subnormální čísla na nulu (FTZ) a používá rychlejší (méně přesné) instrukce pro dělení a odmocniny. Pro raytracer klíčové.


- **`-restrict` (Aliasing)**
- **V Numbě:** *Není potřeba.* Numba interně předpokládá, že se různá pole v paměti nepřekrývají.


- **`--ptxas-options=-v` a další PTX flagy**
- **V Numbě:** `@cuda.jit(ptxas_options=['-v', '-dlcm=cg'])`
- **Funkcionalita:** Umožňuje předat specifické flagy přímo do nízkoúrovňového assembleru (např. vypnutí L1 cache nebo výpis využití paměti do terminálu).

### 2. Typy paměti a funkce

- **`__global__` (Main kernel)**
- **V Numbě:** Dekorátor `@cuda.jit` (bez argumentu `device`).
- **Funkcionalita:** Funkce volaná z CPU, spouští se na GPU v zadané mřížce a blocích.


- **`__device__` (Device funkce)**
- **V Numbě:** Dekorátor `@cuda.jit(device=True)`
- **Funkcionalita:** Pomocná funkce, kterou lze zavolat pouze z jiného GPU kódu (kernelu nebo jiné device funkce).


- **`__shared__` (Sdílená paměť pro blok)**
- **V Numbě:** `s_arr = cuda.shared.array(shape, dtype)` uvnitř kernelu.
- **Funkcionalita:** Velmi rychlá paměť, kterou sdílí všech 32–1024 vláken v rámci jednoho bloku. Slouží jako ručně spravovaná cache.


- **Lokální paměť (Stack, pole pro vlákno)**
- **V Numbě:** `l_arr = cuda.local.array(shape, dtype)` uvnitř kernelu.
- **Funkcionalita:** Privátní pole pro každé vlákno zvlášť. Fyzicky často leží v pomalejší globální paměti (tzv. local memory spill), používá se např. pro lokální zásobník při průchodu stromem (BVH).

### 3. Makra a podmíněný překlad (`#ifdef`, `#define`)

- **V Numbě:** *Není podpora preprocesoru, používá se standardní Python.*
- **Jak to funguje:** Numba vyhodnocuje konstanty v době JIT kompilace. Pokud použijete globální proměnnou (např. `DEBUG_MODE = False`) a uvnitř kernelu napíšete `if DEBUG_MODE:`, Numba provede **dead code elimination**. Kód uvnitř bloku se do výsledného PTX/binárního kódu na GPU vůbec nedostane, stejně jako u `#ifdef` v C++.

***

### MTL → Ray Tracer Mapping

| MTL property | Meaning (MTL)                 | Typical ray tracer interpretation  | Notes / pitfalls                                |
| ------------ | ----------------------------- | ---------------------------------- | ----------------------------------------------- |
| `Kd`         | Diffuse color (albedo)        | `albedo_diffuse`                   | Directly used in Lambertian term                |
| `map_Kd`     | Diffuse texture               | `albedo_texture`                   | Sample instead of constant `Kd`                 |
| `Ks`         | Specular color                | `specular_color` or reflectivity   | Often used as Fresnel base or specular weight   |
| `Ns`         | Specular exponent (shininess) | `roughness` or `phong_exponent`    | Usually converted: `roughness ≈ sqrt(2/(Ns+2))` |
| `Ke`         | Emission color                | `emission` / light source radiance | Non-zero → treat as light                       |
| `Ni`         | Index of refraction           | `ior`                              | Used for refraction (Snell’s law)               |
| `d`          | Opacity (1 = opaque)          | `opacity` / `alpha`                | If `<1`, enable transparency                    |
| `Tf`         | Transmission filter (color)   | `transmittance_color`              | Tints refracted rays                            |
| `illum`      | Illumination model            | shading mode flags                 | Often ignored                                   |

***

#### Critical observations

* `Ns` from Blender can be very large (e.g. 800+)
  → must be remapped to roughness, otherwise highlights become numerically unstable

* `Ks` is **not physically correct reflectivity**
  → treat it as a heuristic weight, not energy-conserving

* `Tf` is often misused
  → if absent, assume white transmission `(1,1,1)`
