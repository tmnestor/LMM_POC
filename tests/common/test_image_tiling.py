"""Tests for common.image_tiling (pure-PIL InternVL dynamic tiling).

tests/ is gitignored — local-only. Locks the tiling contract shared by the HF
preprocessor and the vLLM pre-tiling backend (plans/2026-06-04-adaptive-pre-tiling.md).
"""

from PIL import Image

from common import image_tiling


class TestDynamicPreprocess:
    def test_square_image_single_tile_no_thumbnail(self) -> None:
        # A 1:1 image resolves to a 1x1 grid -> exactly one tile, and the
        # thumbnail is NOT appended when there is only a single tile.
        img = Image.new("RGB", (448, 448))
        tiles = image_tiling.dynamic_preprocess(img, max_num=12, image_size=448, use_thumbnail=True)
        assert len(tiles) == 1

    def test_wide_image_tiles_plus_thumbnail(self) -> None:
        # A 3:1 image with budget >= 3 -> 3 detail tiles + 1 thumbnail = 4.
        img = Image.new("RGB", (1344, 448))
        tiles = image_tiling.dynamic_preprocess(img, max_num=6, image_size=448, use_thumbnail=True)
        assert len(tiles) == 4

    def test_thumbnail_omitted_when_disabled(self) -> None:
        img = Image.new("RGB", (1344, 448))
        tiles = image_tiling.dynamic_preprocess(img, max_num=6, image_size=448, use_thumbnail=False)
        assert len(tiles) == 3

    def test_max_num_caps_tile_count(self) -> None:
        # A very wide image cannot exceed the requested budget in detail tiles.
        img = Image.new("RGB", (448 * 10, 448))
        tiles = image_tiling.dynamic_preprocess(img, max_num=4, image_size=448, use_thumbnail=False)
        assert len(tiles) <= 4

    def test_every_tile_is_image_size_square(self) -> None:
        img = Image.new("RGB", (1344, 448))
        tiles = image_tiling.dynamic_preprocess(img, max_num=6, image_size=448, use_thumbnail=True)
        assert all(t.size == (448, 448) for t in tiles)

    def test_clean_portrait_undertiles_without_min_floor(self) -> None:
        # The headline dense-bank finding: a 2480x3508 A4 portrait (aspect 0.707)
        # matches the 2x3 grid, so raising max_num alone does NOTHING -- it stays
        # at 6 detail tiles whether the ceiling is 12 or 18.
        bank = Image.new("RGB", (2480, 3508))
        for max_num in (12, 18):
            tiles = image_tiling.dynamic_preprocess(
                bank, min_num=1, max_num=max_num, image_size=448, use_thumbnail=True
            )
            assert len(tiles) == 7  # 6 detail + 1 thumbnail, regardless of ceiling

    def test_min_floor_forces_denser_grid_on_portrait(self) -> None:
        # Raising the floor is the only lever that increases the count: min_num=12
        # forces the 3x4 grid -> 12 detail tiles + thumbnail.
        bank = Image.new("RGB", (2480, 3508))
        tiles = image_tiling.dynamic_preprocess(
            bank, min_num=12, max_num=18, image_size=448, use_thumbnail=True
        )
        assert len(tiles) == 13  # 12 detail + 1 thumbnail

    def test_custom_tile_size_respected(self) -> None:
        img = Image.new("RGB", (672, 224))
        tiles = image_tiling.dynamic_preprocess(img, max_num=6, image_size=224, use_thumbnail=False)
        assert all(t.size == (224, 224) for t in tiles)
