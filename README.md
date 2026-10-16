# Steam Style Ingestion

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/license/gpl-3-0)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Steam-Style)](https://github.com/sponsors/Steam-Style)

This is the part of [Steam Style](https://steam.style) that collects every item in the Steam Points Shop and stores it in a [Qdrant](https://qdrant.tech) vector database, so items can be searched by what they look like. The [API](https://github.com/Steam-Style/api) reads from the same database.

## What it does

Once an hour, ingestion pages through the Points Shop and indexes every item that's new or changed since the last run. Unchanged items are skipped, so a run after the first one is quick, and a stopped run carries on where it left off.

For each item it:

- downloads its picture, and for animated items a few frames spread over the video or animation
- embeds the picture with [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224), so text like "rainy city at night" can find it
- describes its colors as a histogram over hue, saturation and brightness, taken from the picture and its frames together
- measures how much of it has no color to speak of (greys and near-black), since the color histogram leaves those out
- stores Steam's details about it: name, category, price, assets, game, links and dates
- for game profiles, stores the theme colors and the parts that come with it

Items that drop out of the Points Shop are kept and marked as no longer sold, and items that are only sold in a bundle are marked as well.

### What's stored for each item

Each point's ID is the item's Points Shop ID. It has three vectors:

| Vector   | Size               | What it is                                                                            |
| -------- | ------------------ | ------------------------------------------------------------------------------------- |
| `image`  | 768                | SigLIP 2 embedding of the item's picture                                              |
| `color`  | 216                | Color histogram over 24 hues, 3 saturations and 3 brightnesses                        |
| `frames` | 768 for each frame | SigLIP 2 embedding of each frame of animated items, and of the picture for still ones |

The payload has the item (`item.*`), its game (`app.*`), links to Steam (`urls.*`) and dates (`timestamps.*`), plus `profile` for game profiles. `item.neutral` is the share of the picture without color, from 0 to 1. `item.popularity` is the item's place in the Points Shop's popular order, 1 being the most popular, updated once a day.

## Running

You need Docker with Docker Compose.

```bash
git clone https://github.com/Steam-Style/ingestion.git
cd ingestion
echo "DATABASE_URL=http://qdrant:6333" > .env
docker compose up -d
```

This starts Qdrant and the ingestion service. Qdrant is only reachable from the machine itself, on ports 6333 and 6334. The first run indexes the whole Points Shop, which takes a while, mostly downloading pictures and videos. The embedding model is downloaded on first start and kept in a volume.

To run it without Docker, against a Qdrant you already have, set `DATABASE_URL=http://localhost:6333` in `.env` and run:

```bash
uv sync
uv run --env-file .env python src/main.py
```

### Settings

Set these as environment variables.

| Variable                     | Default                                     | What it's for                                              |
| ---------------------------- | ------------------------------------------- | ---------------------------------------------------------- |
| `DATABASE_URL`               | required                                    | Where Qdrant is                                            |
| `COLLECTION_NAME`            | `Steam-Style-Items`                         | The collection to fill, created if it doesn't exist        |
| `DEVICE`                     | `cpu`                                       | Where the model runs, like `cuda`                          |
| `MODEL_NAME`                 | `google/siglip2-base-patch16-224`           | The embedding model, must match the API's                  |
| `INGESTION_INTERVAL_SECONDS` | `3600`                                      | Time between runs                                          |
| `POPULARITY_INTERVAL_SECONDS` | `86400`                                   | Time between updates of the popular order                  |
| `PROFILE_PREVIEW_URL`        | `https://steamcommunity.com/id/MayorCarol/` | A public Steam profile used to preview game profile themes |
| `IMAGE_DOWNLOAD_WORKERS`     | `12`                                        | Pictures downloaded at the same time                       |
| `FRAMES_PER_ITEM`            | `8`                                         | Frames taken from each animated item                       |

### Items Steam no longer lists

The Points Shop API only lists items that are still sold. If you want older items too, `src/backfill_archive.py` adds them from Points Shop responses the Internet Archive saved, and marks them as no longer sold. Run it once where ingestion runs:

```bash
python src/backfill_archive.py --dry-run
python src/backfill_archive.py
```

## The embeddings package

`packages/steam-style-embeddings` has the SigLIP 2 and color embedders. The API installs it from this repository too, so text searches and stored pictures are embedded the same way.

---

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/Steam-Style/ingestion?style=social)](https://github.com/Steam-Style/ingestion/stargazers)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Steam-Style?style=social)](https://github.com/sponsors/Steam-Style)

Made with ❤️ by the Steam Style team

[Report an Issue](https://github.com/Steam-Style/ingestion/issues) • [Visit Steam Style](https://steam.style)

</div>
