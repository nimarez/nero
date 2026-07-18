# Main room assets

Gaussian-splat scene with a matching collision mesh. Source capture:
*"Industrial-style workshop with wooden beams"*.

| File | Size | Purpose |
|---|---|---|
| `assets/main_room.ply` | ~187 MB | Gaussian splat — **visual** representation |
| `assets/main_room_collider.glb` | ~4.2 MB | Simplified mesh — **collision / physics** |

Two representations of the same room: the splat is what you *see*, the collider is
what you *hit*. Splats carry no usable collision geometry, hence the separate `.glb`.

## Integration status

These are reference assets, not an installed Booster Studio scene. The current
`nero-setup-booster-room` command installs the smaller CC0 living room described
in the [root README](../../../../../README.md#furnished-living-room); it does not
load this splat or collider. Integration requires a PLY splat renderer, a
simulator-compatible collider reference or conversion, and an explicit frame
alignment check.

## Getting the files (Git LFS)

Both assets are stored in Git LFS — `main_room.ply` is well over GitHub's 100 MB
per-file limit. Without LFS you get ~130-byte text pointers instead of geometry,
which fail later with a confusing parse error.

```bash
git lfs install     # one-time, per machine
git lfs pull        # if you cloned before installing LFS
```

Verify you got the real thing, not a pointer:

```bash
ls -lh assets/main_room.ply
# expected: approximately 187 MiB, not approximately 130 bytes

head -1 assets/main_room.ply
# "version https://git-lfs.github.com/spec/v1" means it is still a pointer
```

See [`../industrial_storage_room/README.md`](../industrial_storage_room/README.md)
for fuller LFS notes, including `GIT_LFS_SKIP_SMUDGE=1` for CI clones that don't
need geometry.

## Notes

- **Storage** — this scene adds about 191 MiB. Together with the industrial room,
  a full LFS checkout transfers roughly 325 MiB. Check the repository host's
  current quota before enabling these downloads in CI.
- **Coordinate frames** — splat and collider are exported from the same source and
  share an origin. If one appears offset from the other, suspect an up-axis
  convention (Y-up vs Z-up) rather than a bad export.
