# Industrial storage room assets

Gaussian-splat capture with a matching simplified collision mesh.

| File | Size | Purpose |
|---|---|---|
| `assets/industrial_storage_room.ply` | ~125 MB | Gaussian splat — **visual** representation |
| `assets/industrial_storage_room_collider.glb` | ~8.5 MB | Simplified mesh — **collision / physics** |

The splat is the visual representation; the GLB is the physics representation.
Splats have no usable collision geometry, hence the separate mesh.

## Integration status

These are reference assets, not an installed Booster Studio scene. The current
`nero-setup-booster-room` command installs the smaller CC0 living room described
in the [root README](../../../../../README.md#furnished-living-room); it does not
load this splat or collider. A future scene integration must provide a renderer
for the PLY splat, convert or reference the collider in the simulator's supported
format, and verify that both assets share the K1/world coordinate frame.

---

## Git LFS

They are **not** in the repository as ordinary files. `industrial_storage_room.ply`
is 125 MB, over GitHub's hard 100 MB per-file limit, so both assets are tracked by
[Git LFS](https://git-lfs.com) via the repo's `.gitattributes`.

### Download

Install Git LFS once per machine:

```bash
brew install git-lfs
# or, on Debian/Ubuntu:
sudo apt install git-lfs

git lfs install          # one-time, per user
```

Then either clone fresh:

```bash
git clone git@github.com:nimarez/nero.git      # LFS files download automatically
```

Or, if you cloned before installing LFS:

```bash
git lfs pull             # fetches the real file contents
```

### Detect an unresolved pointer

If Git LFS isn't installed, the clone still "succeeds" — but these files arrive as
**~130-byte text pointers**, not geometry. Loading one will fail with a confusing
parse error rather than anything mentioning LFS.

Spot it by size, or by looking at the file:

```bash
ls -lh assets/industrial_storage_room.ply
# expected: approximately 125 MiB, not approximately 130 bytes

head -1 assets/industrial_storage_room.ply
# "version https://git-lfs.github.com/spec/v1" means it is still a pointer
```

The fix is always the same: `git lfs install && git lfs pull`.

## Notes

- **Storage** — these assets consume about 134 MiB and every full LFS checkout
  transfers them. Check the repository host's current storage and bandwidth quota
  before enabling LFS downloads in CI.
- **Shallow/CI clones** — set `GIT_LFS_SKIP_SMUDGE=1` to clone pointers only, for
  jobs that don't need the geometry:
  ```bash
  GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:nimarez/nero.git
  ```
- **Coordinate frames** — the assets were exported from the same source and share
  an origin. Validate scale, handedness, and Y-up versus Z-up after conversion.
