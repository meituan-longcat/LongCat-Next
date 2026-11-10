# Paper Figures

Figures from **dNaViT: Discrete Codes are General-Purpose Visual Representations**, by the Meituan LongCat Team.

The assets match manuscript source revision `6dd463749ad10e3fbcc9f3f2e830d53cc8bfa5ab`, retrieved on 2026-11-09. Figure numbers follow the numbered captions in the manuscript. The qualitative showcase page now has a numbered caption (Figure 2), so the architecture, tokenization, and Refiner figures have moved from 2/3/5 to 3/4/6.

| Paper figure | Source asset | README asset | Placement |
| :--- | :--- | :--- | :--- |
| Figure 1 | `figures/teaser.pdf` | [PNG](figure-1-teaser.png) / [PDF](figure-1-teaser.pdf) | Model introduction |
| Figure 2 | `figures/case_wall-v15.pdf` | [PNG](figure-2-showcase.png) / [PDF](figure-2-showcase.pdf) | Collapsible downstream showcase |
| Figure 3 | `figures/dnavit.pdf` | [PNG](figure-3-architecture.png) / [PDF](figure-3-architecture.pdf) | Semantic features and residual modulation |
| Figure 4 | `figures/group_tokenizer.pdf` | [PNG](figure-4-tokenization.png) / [PDF](figure-4-tokenization.pdf) | Grouped product quantization |
| Figure 6 | `figures/refiner.pdf` | [PNG](figure-6-refiner.png) / [PDF](figure-6-refiner.pdf) | Code-conditioned Refiner |

Figure 5 (dual-branch qualitative comparison) is not embedded in this README; the selected figures retain their manuscript numbers rather than being renumbered consecutively.

The PDFs are unchanged copies of the manuscript assets. The PNG previews are rendered directly from the single-page PDFs with Poppler at a maximum dimension of 2400 pixels, preserving the aspect ratio and figure content:

```bash
pdftoppm -f 1 -singlefile -scale-to 2400 -png figure-1-teaser.pdf figure-1-teaser
pdftoppm -f 1 -singlefile -scale-to 2400 -png figure-2-showcase.pdf figure-2-showcase
pdftoppm -f 1 -singlefile -scale-to 2400 -png figure-3-architecture.pdf figure-3-architecture
pdftoppm -f 1 -singlefile -scale-to 2400 -png figure-4-tokenization.pdf figure-4-tokenization
pdftoppm -f 1 -singlefile -scale-to 2400 -png figure-6-refiner.pdf figure-6-refiner
```

The figures illustrate the paper's full methodology, including downstream models and the Refiner. Refiner inference scripts are provided on Hugging Face.

## Source Integrity

Each repository PDF is byte-identical to its manuscript source asset. SHA-256:

```text
26b77a7047d65b4a8be9178ab8aa3fabaaf5aaa7dc4d5f4a125bb31ee2bdd98a  figure-1-teaser.pdf
f27b2512cc8198abc589d53401397078fd6fc305b55af35c08052138c9c62ba0  figure-2-showcase.pdf
c4336abcb4c3a3b039e18948081fdea5fa6243fa2f7c6152062472ea1b77edb1  figure-3-architecture.pdf
59bfae32ce87c8171c94c652f11f0a79f71510cce8d5be9cd4cff4b577a2e0b2  figure-4-tokenization.pdf
8c622c6d5abbee93b900e4f43535e7c8c089a497e71c46be789e408878c2cd9f  figure-6-refiner.pdf
```
