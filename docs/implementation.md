# Implementation notes

STC runs once before LLM prefill and preserves every text token. Video tokens
are consolidated by local spatial regions and adjacent-frame reuse; adjacent
audio pairs above the configured similarity threshold are replaced by their
mean.

SSPA runs after native prefill. It stores historical keys in INT4 and assigns
value pages to BF16, INT4, or INT2. Newly generated states enter Recent-FP and
are recompressed in aligned 128-token chunks. The fused Triton decoder consumes
the mixed cache directly.

The release implements the STC/SSPA inference path and packed kernels with
portable model and dataset interfaces. Method profiles are stored explicitly in
configs/. Internal sweeps, private review workflows, historical result-migration
tools and cluster launch files are excluded.

For MiniCPM-o-4.5, STC is applied to the final encoder embeddings at the
post-encoder/pre-LLM boundary. MiniCPM image blocks are treated as their native
8-by-8 visual grids, and audio/video bounds provide the modality map used by
SSPA. No MiniCPM encoder or prompt code is replaced.
