# CTU FIT NI-PG1 Matej Zeman (zemanm40) 2026

My Sara raytracer in numba-cuda.

## Structure/Installation

Separate homeworks are in the branches (hw0x). The main source code is inside `src/` folder. It is using my `utils/` module.

For Python package on local machine, I use conda venv called 'raytracer'. Use `run_local.sh` script to run specific branch.

I managed to run it on FIT cluster as well with using standard `.venv/` installed from `requirements.txt`
If you do not have conda package manager, build your `.venv` form `requirements.txt`.


The output will be in the `src/output/` folder as `output.ppm/png`.

### TinyObjLoader

I use the c++ header tiny-obj-loader library together with python bindings. I parsed the triangles there too. So user on different machine needs to recompile it for there is older version of python on the cluster or your device.

Use `./rebuild_tinyobjloader.sh python3.<version>` for compilation from the root directory. Note: For each homework branch may be different version of the library. So recompiling when switching branches is adviced. You can specify your version of python to compile for, but default is `python3.14`.

## Render Times

Testing on my box-advanced with +-5000 triangles.
The meassurements assume the data is already on gpu.

> Dimensions per machine:
> - 1440**2 on local rtx3070 mobile.
> - 5760**2 on remote A100 on cluster.


## Homework render times:
Pure rendering on both machines:
| hw  | local  | remote | note    |
| --- | ------ | ------ | ------- |
| 1   | 2.26 s | 1.18 s | cook    |
| 2   | x.xx s | x.xx s | SAH BVH |
| 3   | x.xx s | x.xx s |         |
| 4   | x.xx s | x.xx s |         |
| 5   | x.xx s | x.xx s |         |

## Homework renders

<img src="hw01/output/output.png" width="25%"/>
<img src="hw02/output/output.png" width="25%"/>

***

<img src="hw03/output/output.png" width="25%"/>
<img src="hw04/output/output.png" width="25%"/>

***

<img src="hw05/output/output.png" width="25%"/>


## Differences with C++ CUDA (in czech)

Shrnutí převodu konceptů z CUDA C++ (`nvcc`) do Numby, ideální pro zkopírování na později.

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

***

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


***


### 3. Makra a podmíněný překlad (`#ifdef`, `#define`)

- **V Numbě:** *Není podpora preprocesoru, používá se standardní Python.*
- **Jak to funguje:** Numba vyhodnocuje konstanty v době JIT kompilace. Pokud použijete globální proměnnou (např. `DEBUG_MODE = False`) a uvnitř kernelu napíšete `if DEBUG_MODE:`, Numba provede **dead code elimination**. Kód uvnitř bloku se do výsledného PTX/binárního kódu na GPU vůbec nedostane, stejně jako u `#ifdef` v C++.

