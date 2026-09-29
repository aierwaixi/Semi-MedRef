# PosMed preprocessing artifact

`brain_mri_slice_to_pid.csv` freezes the public Cheng Brain MRI patient IDs and
five-fold assignments used by `scripts/prepare_posmed_splits.py`.
`brain_mri_slice_to_pid.provenance.json` records the public source revision,
checksums, projection, and validation against the original `cvind.mat` file.

`r0.05/` records the parent labeled split, while `r0.15/` contains the exact
labeled, unlabeled, validation, and test manifests used by the cross-domain
experiment. The manifests contain no machine-local paths: images and masks are
resolved from the public PosMed directory passed through `--data-root`. See the
package-level `README.md` for regeneration commands.
