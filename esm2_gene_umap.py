"""
esm2_gene_umap.py
=================
Explore the ESM2 gene-symbol embedding file and draw a color-coded UMAP.

INPUT
-----
Homo_sapiens_GRCh38_gene_symbol_to_embedding_ESM2.pt
    A dict:  gene symbol (str) -> 5120-dim float32 vector (ESM2 protein embedding).
    (19,790 genes in the file you uploaded.)

OUTPUT (written to --out, default ./results)
------
    esm2_gene_summary.txt        basic stats about the file
    esm2_gene_umap.png           UMAP colored by gene family
    esm2_gene_umap_coords.csv    gene, family, umap1, umap2  (reuse for other figures)
    esm2_family_knn_purity.csv   how well each family clusters (see step 6)

HOW TO RUN (Anaconda Prompt on Windows)
---------------------------------------
    conda create -n mapumap python=3.10 -y
    conda activate mapumap
    pip install numpy pandas matplotlib scikit-learn umap-learn
    (put the .pt file in the same folder as this script first)
    python esm2_gene_umap.py
    # or, if the .pt file is somewhere else:
    python esm2_gene_umap.py --pt "C:\\path\\to\\Homo_sapiens_GRCh38_gene_symbol_to_embedding_ESM2.pt"

    (torch is OPTIONAL. If it is installed it is used to read the file;
     if not, a built-in torch-free reader is used.)
"""

import argparse
import collections
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# Name of the embedding file, expected in the same folder as this script
DEFAULT_PT = str(Path(__file__).resolve().parent /
                 "Homo_sapiens_GRCh38_gene_symbol_to_embedding_ESM2.pt")


# ---------------------------------------------------------------------------
# STEP 1. Load the .pt file
# ---------------------------------------------------------------------------
def load_embeddings_torch_free(pt_path):
    """Read a torch-saved dict of tensors WITHOUT importing torch.

    A .pt file is a zip archive containing:
      * data.pkl  - a pickle describing the dict (gene -> tensor "recipe")
      * data/<k>  - the raw float32 numbers for tensor number k
    We unpickle data.pkl with a stub that records each recipe instead of
    building a real tensor, then read the raw bytes with numpy.
    """

    class _Storage:                      # remembers which data/<k> file to read
        def __init__(self, key):
            self.key = key

    class _FloatStorage:                 # placeholder for torch.FloatStorage
        pass

    def _rebuild_tensor(storage, offset, size, stride, *_):
        return (storage, offset, tuple(size), tuple(stride))

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor
            if name == "FloatStorage":
                return _FloatStorage
            if name == "OrderedDict":
                return collections.OrderedDict
            return super().find_class(module, name)

        def persistent_load(self, pid):  # pid = ('storage', type, key, loc, n)
            return _Storage(pid[2])

    with zipfile.ZipFile(pt_path) as z:
        names = z.namelist()
        root = names[0].split("/")[0]                      # usually "archive"
        recipes = _Unpickler(z.open(f"{root}/data.pkl")).load()
        genes = list(recipes.keys())
        dim = recipes[genes[0]][2][0]
        mat = np.empty((len(genes), dim), dtype=np.float32)
        for i, g in enumerate(genes):
            storage, offset, size, _ = recipes[g]
            raw = np.frombuffer(z.read(f"{root}/data/{storage.key}"), dtype=np.float32)
            mat[i] = raw[offset: offset + size[0]]
    return genes, mat


def load_embeddings(pt_path):
    """Return (list_of_gene_symbols, matrix[n_genes, dim])."""
    try:
        import torch  # optional
        d = torch.load(pt_path, map_location="cpu")
        genes = list(d.keys())
        mat = np.stack([d[g].float().numpy() for g in genes])
        return genes, mat
    except ImportError:
        return load_embeddings_torch_free(pt_path)


# ---------------------------------------------------------------------------
# STEP 2. Label genes with a "family" (the color code)
# ---------------------------------------------------------------------------
# The file has NO labels, so families are guessed from gene-symbol prefixes
# (e.g. RPL/RPS = ribosomal proteins). Order matters: first match wins.
# These are a quick HEURISTIC, not a curated annotation. For the real
# project, replace with GO / Reactome / KEGG or your MOA table.
FAMILY_RULES = [
    ("Ribosomal",          r"^(RPL|RPS|MRPL|MRPS)\d"),
    ("Histone",            r"^(HIST|H1-|H2AC|H2BC|H3C|H4C|H2AZ|H3-)"),
    ("Olfactory receptor", r"^OR\d"),
    ("Taste receptor",     r"^TAS[12]R"),
    ("Zinc finger",        r"^(ZNF|ZSCAN|ZBTB|ZFP)\d*"),
    ("Keratin",            r"^(KRT|KRTAP)\d"),
    ("Solute carrier",     r"^SLC\d"),
    ("Collagen",           r"^COL\d"),
    ("Cytochrome P450",    r"^CYP\d"),
    ("HLA / immune MHC",   r"^HLA-"),
    ("Immunoglobulin",     r"^IG[HKL][VDJCAGEM]"),
    ("Cadherin / PCDH",    r"^(CDH|PCDH)\d"),
    ("Chemokine / IL",     r"^(CCL|CXCL|IL\d)"),
    ("Mito complex I",     r"^NDUF"),
    ("Proteasome",         r"^PSM[ABCD]\d"),
    ("Translation init.",  r"^EIF\d"),
    ("Tubulin",            r"^TUB[AB]"),
]
_COMPILED = [(name, re.compile(rx)) for name, rx in FAMILY_RULES]


def assign_family(symbol):
    for name, rx in _COMPILED:
        if rx.match(symbol):
            return name
    return "Other"


# ---------------------------------------------------------------------------
# STEP 3. Summary of the file (what information is in it?)
# ---------------------------------------------------------------------------
def summarize(genes, mat, fams, out_path):
    norms = np.linalg.norm(mat, axis=1)
    lines = [
        f"genes                 : {len(genes)}",
        f"embedding dimension   : {mat.shape[1]}",
        f"NaN values            : {int(np.isnan(mat).sum())}",
        f"all-zero vectors      : {int((norms == 0).sum())}",
        f"vector L2 norm        : min {norms.min():.2f}  median {np.median(norms):.2f}  max {norms.max():.2f}",
        f"example genes         : {', '.join(genes[:8])} ...",
        "",
        "genes per heuristic family:",
    ]
    for fam, n in collections.Counter(fams).most_common():
        lines.append(f"  {fam:<22}{n}")
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(text)


# ---------------------------------------------------------------------------
# STEP 4. Reduce dimensions: L2-normalize -> PCA(50) -> UMAP(2)
# ---------------------------------------------------------------------------
def embed_2d(mat, n_pcs=50, n_neighbors=30, min_dist=0.3, seed=0):
    from sklearn.decomposition import PCA
    import umap

    # L2-normalize so Euclidean distance ~ cosine distance
    x = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8, None)
    # PCA first: 5120 dims is slow/noisy for UMAP; 50 PCs keeps the structure
    pcs = PCA(n_components=n_pcs, random_state=seed).fit_transform(x)
    # UMAP: each dot = one gene; close dots = similar embeddings
    xy = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
        random_state=seed,
    ).fit_transform(pcs)
    return pcs, xy


# ---------------------------------------------------------------------------
# STEP 5. Plot (dark style, same look as the MCF7 MOA figure)
# ---------------------------------------------------------------------------
def plot_umap(xy, fams, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = np.asarray(fams)
    order = [f for f, _ in collections.Counter(fams).most_common() if f != "Other"]
    cmap = plt.get_cmap("tab20")
    colors = {f: cmap(i % 20) for i, f in enumerate(order)}

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#101116")
    ax.set_facecolor("#1a1b24")

    # background: unlabeled genes, small + faint so labeled families stand out
    m = fams == "Other"
    ax.scatter(xy[m, 0], xy[m, 1], s=2, c="#555a6b", alpha=0.35,
               linewidths=0, label=f"Other (n={m.sum()})", rasterized=True)
    # foreground: one color per family
    for f in order:
        m = fams == f
        ax.scatter(xy[m, 0], xy[m, 1], s=9, color=colors[f], alpha=0.9,
                   linewidths=0, label=f"{f} (n={m.sum()})")

    ax.set_title("UMAP - ESM2 gene embeddings by gene family",
                 color="white", fontweight="bold")
    ax.set_xlabel("UMAP 1", color="white")
    ax.set_ylabel("UMAP 2", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#444")
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                    fontsize=8, frameon=False, markerscale=2)
    for t in leg.get_texts():
        t.set_color("white")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# STEP 6. Quantify clustering: k-nearest-neighbor purity
# ---------------------------------------------------------------------------
def knn_purity(pcs, fams, k=15):
    """For each gene in a family, what fraction of its k nearest neighbors
    (cosine, in PCA space) are in the SAME family?  Compare with the
    'chance' level = that family's share of all genes.
    This is the same kind of test as your hypothesis (similar function ->
    close in the embedding), but with a number instead of just a picture."""
    from sklearn.neighbors import NearestNeighbors

    fams = np.asarray(fams)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(pcs)
    idx = nn.kneighbors(pcs, return_distance=False)[:, 1:]   # drop self
    same = fams[idx] == fams[:, None]
    rows = []
    for f in sorted(set(fams) - {"Other"}):
        m = fams == f
        rows.append({
            "family": f,
            "n_genes": int(m.sum()),
            f"knn_purity_k{k}": round(float(same[m].mean()), 3),
            "chance": round(float(m.mean()), 4),
        })
    return pd.DataFrame(rows).sort_values(f"knn_purity_k{k}", ascending=False)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Default: look for the .pt file in the same folder as this script, so
    # you can simply run:  python esm2_gene_umap.py
    ap.add_argument("--pt", default=DEFAULT_PT,
                    help="path to the ESM2 .pt file (default: next to this script)")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not Path(args.pt).exists():
        raise SystemExit(
            f"\nCould not find the embedding file:\n  {args.pt}\n"
            "Put Homo_sapiens_GRCh38_gene_symbol_to_embedding_ESM2.pt in the same "
            "folder as this script, or run:\n"
            '  python esm2_gene_umap.py --pt "C:\\path\\to\\file.pt"\n')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1/6 loading embeddings ...")
    genes, mat = load_embeddings(args.pt)
    fams = [assign_family(g) for g in genes]

    print("2/6 summary ...")
    summarize(genes, mat, fams, out / "esm2_gene_summary.txt")

    print("3/6 PCA + UMAP (a few minutes) ...")
    pcs, xy = embed_2d(mat, seed=args.seed)

    print("4/6 saving coordinates ...")
    pd.DataFrame({"gene": genes, "family": fams,
                  "umap1": xy[:, 0], "umap2": xy[:, 1]}
                 ).to_csv(out / "esm2_gene_umap_coords.csv", index=False)

    print("5/6 plotting ...")
    plot_umap(xy, fams, out / "esm2_gene_umap.png")

    print("6/6 kNN purity ...")
    pur = knn_purity(pcs, fams)
    pur.to_csv(out / "esm2_family_knn_purity.csv", index=False)
    print(pur.to_string(index=False))
    print(f"\nDone. Files are in: {out.resolve()}")


if __name__ == "__main__":
    main()
