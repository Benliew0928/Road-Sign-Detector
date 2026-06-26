from pathlib import Path

from PIL import Image

from roadsign_assist.datasets.contact_sheet import ReviewTile, render_contact_sheet


def test_contact_sheet_has_stable_grid_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 60), color=(220, 40, 40)).save(source)
    output = tmp_path / "sheet.jpg"
    render_contact_sheet(
        [ReviewTile(label=f"sign_{index:03d}", image_path=source) for index in range(5)],
        output,
        columns=3,
        tile_width=120,
        tile_height=100,
    )
    with Image.open(output) as sheet:
        assert sheet.size == (360, 200)
