#!/usr/bin/env python3
"""Render a .vmap with local Heroes III artwork, without starting VCMI."""
import argparse
from functools import cache
import json
import re
import struct
import zlib
from pathlib import Path
from zipfile import BadZipFile, ZipFile

TILE_SIZE = 32
TERRAIN = dict(zip(
    ("dt", "sa", "gr", "sn", "sw", "rg", "sb", "lv", "wt", "rc"),
    ("dirttl", "sandtl", "grastl", "snowtl", "swmptl", "rougtl", "subbtl", "lavatl", "watrtl", "rocktl"),
))
ROADS = {"pd": "dirtrd", "pg": "gravrd", "pc": "cobbrd"}
RIVERS = {"rw": "clrrvr", "ri": "icyrvr", "rm": "mudrvr", "rl": "lavrvr"}
COLORS = ("red", "blue", "tan", "green", "orange", "purple", "teal", "pink")
TILE_LAYERS = re.compile(r"([a-z]{2})(\d+)([_|+\-])")


def lod_index(data_dir):
    entries = {}
    archives = sorted(data_dir.glob("*.[lL][oO][dD]"), key=lambda p: "h3ab_" in p.name.lower())
    if not archives:
        raise ValueError(f"No .lod files in {data_dir}")
    for path in archives:
        with path.open("rb") as stream:
            magic, _, count = struct.unpack("<4sII", stream.read(12))
            if magic != b"LOD\0":
                raise ValueError(f"Invalid LOD header in {path}")
            stream.seek(92)
            for _ in range(count):
                name, offset, size, _, compressed = struct.unpack("<16sIIII", stream.read(32))
                name = name.split(b"\0", 1)[0].lower()
                if name.endswith(b".def"):
                    entries[name.decode("ascii")] = path, offset, size, compressed
    return entries


def decode_frame(data, offset):
    # DEF layouts 0–3 follow the pinned VCMI CDefFile format.
    _, mode, full_w, full_h, width, height, left, top = struct.unpack_from("<6I2i", data, offset)
    base = offset + 32
    if mode == 1 and width > full_w and height > full_h:
        base -= 16
        width, height, left, top = full_w, full_h, 0, 0
    if mode == 0:
        return (full_w, full_h, left, top, width, height), data[base:base + width * height]
    if mode not in (1, 2, 3):
        raise ValueError(f"Unsupported DEF pixel format {mode}")
    pixels = bytearray()
    cursor = base + struct.unpack_from("<H", data, base)[0] if mode == 2 else base
    for row in range(height):
        if mode == 1:
            cursor = base + struct.unpack_from("<I", data, base + row * 4)[0]
        elif mode == 3:
            cursor = base + struct.unpack_from("<H", data, base + row * 2 * (width // 32))[0]
        row_end = (row + 1) * width
        while len(pixels) < row_end:
            if mode == 1:
                code, length = data[cursor], data[cursor + 1] + 1
                cursor += 2
                literal = code == 255
            else:
                code, remainder = divmod(data[cursor], 32)
                length, literal = remainder + 1, code == 7
                cursor += 1
            if literal:
                pixels.extend(data[cursor:cursor + length])
                cursor += length
            else:
                pixels.extend(bytes([code]) * length)
        if len(pixels) != row_end:
            raise ValueError("DEF pixel run crosses a row")
    return (full_w, full_h, left, top, width, height), bytes(pixels)


def render(terrain, objects, data_dir):
    from PIL import Image

    entries = lod_index(data_dir)

    @cache
    def animation(name):
        filename = name.lower().removesuffix(".def") + ".def"
        if filename not in entries:
            raise ValueError(f"Sprite {filename} is missing from the installed LOD files")
        path, offset, size, compressed = entries[filename]
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(compressed or size)
        data = zlib.decompress(raw) if compressed else raw
        groups = {}
        cursor = 784  # 16-byte header and 256 RGB palette entries
        for _ in range(struct.unpack_from("<I", data, 12)[0]):
            group, count = struct.unpack_from("<II", data, cursor)
            cursor += 16 + count * 13
            groups[group] = struct.unpack_from(f"<{count}I", data, cursor)
            cursor += count * 4
        return data, groups

    @cache
    def sprite(name, frame=0, flip="_", opaque=False):
        data, groups = animation(name)
        offsets = groups[0] if 0 in groups else next(iter(groups.values()))
        (full_w, full_h, left, top, width, height), pixels = decode_frame(data, offsets[frame])
        image = Image.frombytes("P", (width, height), pixels)
        palette = bytearray(data[16:784])
        if not opaque:
            palette[:24] = bytes(24)
            image.info["transparency"] = bytes([0, 32, 64, 128, 128, 0, 0, 0] + [255] * 248)
        image.putpalette(palette)
        full = Image.new("RGBA", (full_w, full_h))
        full.alpha_composite(image.convert("RGBA"), (left, top))
        if flip in "-+":
            full = full.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if flip in "|+":
            full = full.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return full

    canvas = Image.new("RGBA", (len(terrain[0]) * TILE_SIZE, len(terrain) * TILE_SIZE))
    overlays = []
    for y, row in enumerate(terrain):
        for x, tile in enumerate(row):
            layers = TILE_LAYERS.findall(tile)
            if not layers or "".join("".join(layer) for layer in layers) != tile:
                raise ValueError(f"Invalid terrain tile {tile!r}")
            code, frame, flip = layers[0]
            canvas.alpha_composite(sprite(TERRAIN[code], int(frame), flip, True), (x * TILE_SIZE, y * TILE_SIZE))
            overlays.extend((code, int(frame), flip, x, y) for code, frame, flip in layers[1:])
    for code, frame, flip, x, y in sorted(overlays, key=lambda layer: layer[0] in ROADS):
        file = (ROADS if code in ROADS else RIVERS)[code]
        canvas.alpha_composite(sprite(file, frame, flip),
                               (x * TILE_SIZE, y * TILE_SIZE + (16 if code in ROADS else 0)))

    for obj in sorted(objects, key=lambda item: (item["y"], item["type"] == "hero", item["x"])):
        name = obj["template"]["animation"]
        names = [name]
        owner = obj.get("options", {}).get("owner")
        if obj["type"] == "hero" and owner in COLORS:
            names.append(f"af{COLORS.index(owner):02d}")
        for name in names:
            image = sprite(name)
            canvas.alpha_composite(image, ((obj["x"] + 1) * TILE_SIZE - image.width,
                                           (obj["y"] + 1) * TILE_SIZE - image.height))
    return canvas.convert("RGB")


def installed_data():
    return Path.home() / "Library/Application Support/vcmi/Data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path)
    parser.add_argument("png", type=Path)
    parser.add_argument("--data-dir", type=Path, default=installed_data(), help="directory containing original .lod files")
    parser.add_argument("--level", choices=("surface", "underground"), default="surface")
    args = parser.parse_args()
    if args.png.resolve() == args.map.resolve() or args.png.suffix.lower() != ".png":
        parser.error("output must be a separate .png file")
    try:
        with ZipFile(args.map) as archive:
            level = json.loads(archive.read("header.json"))["mapLevels"][args.level]
            terrain = json.loads(archive.read(f"{args.level}_terrain.json"))
            objects = json.loads(archive.read("objects.json"))
        if not terrain or len(terrain) != level["height"] or any(len(row) != level["width"] for row in terrain):
            parser.error("terrain dimensions differ from the map header")
        render(terrain, [obj for obj in objects if obj["l"] == level["index"]], args.data_dir).save(args.png)
    except ModuleNotFoundError:
        parser.error("Pillow is required. Use a Python environment with Pillow installed.")
    except (BadZipFile, OSError, ValueError, KeyError, TypeError, IndexError, struct.error, zlib.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
