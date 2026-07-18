# Industrial storage room

Gaussian-splat scene with a matching collision mesh.

| File | Size | Purpose |
|---|---|---|
| `assets/industrial_storage_room.ply` | ~125 MB | Gaussian splat — **visual** representation |
| `assets/industrial_storage_room_collider.glb` | ~8.5 MB | Simplified mesh — **collision / physics** |

Two representations of the same room: the splat is what you *see*, the collider is
what you *hit*. Splats have no usable collision geometry, hence the separate `.glb`.

---

## These files are stored in Git LFS

They are **not** in the repository as ordinary files. `industrial_storage_room.ply`
is 125 MB, over GitHub's hard 100 MB per-file limit, so both assets are tracked by
[Git LFS](https://git-lfs.com) via the repo's `.gitattributes`.

### Getting them

Install Git LFS once per machine:

```bash
# macOS
brew install git-lfs
# Debian/Ubuntu
sudo apt install git-lfs

git lfs install          # one-time, per user
```

Then either clone fresh:

```bash
git clone git@github.com:nimarez/nero.git      # LFS files download automatically
```

…or, if you already cloned **before** installing LFS:

```bash
git lfs pull             # fetches the real file contents
```

### …or else: what you get without LFS

If Git LFS isn't installed, the clone still "succeeds" — but these files arrive as
**~130-byte text pointers**, not geometry. Loading one will fail with a confusing
parse error rather than anything mentioning LFS.

Spot it by size, or by looking at the file:

```bash
$ ls -l assets/industrial_storage_room.ply
-rw-r--r--  1 you  staff  132 ...          # ← 132 bytes, not 125 MB

$ head -1 assets/industrial_storage_room.ply
version https://git-lfs.github.com/spec/v1  # ← a pointer, not a point cloud
```

The fix is always the same: `git lfs install && git lfs pull`.

## Notes

- **Quota** — these two assets consume ~134 MB of the repo's LFS storage, and LFS
  bandwidth is billed per *download*, so every fresh clone and CI run pulls that
  again. GitHub's free tier is 1 GB storage + 1 GB bandwidth/month. Keep an eye on
  it before adding many more scenes.
- **Shallow/CI clones** — set `GIT_LFS_SKIP_SMUDGE=1` to clone pointers only, for
  jobs that don't need the geometry:
  ```bash
  GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:nimarez/nero.git
  ```
- **Coordinate frames** — the splat and collider are exported from the same source
  and share an origin; if one appears offset from the other, suspect an
  up-axis convention (Y-up vs Z-up) rather than a bad export.
