# Steam Style Item Ingestion

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/license/gpl-3-0)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Steam-Style)](https://github.com/sponsors/Steam-Style)

This repository contains the ingestion portion of the [Steam Style project](https://www.steam.style), which is designed to collect and store data about items available in the Steam Points Shop. The data can then be easily retrieved and queried using the API portion of the project, which is available in [a separate repository](https://github.com/Steam-Style/api).

The ingestion process periodically fetches items from the Steam Points Shop using the official Steam API and stores the results in a local Qdrant vector database. The purpose of the project is to offer a more convenient way of finding relevant items for profile customization, as an alternative to Steam's own lackluster search features.

## Running

### Prerequisites

- Git
- Docker + Docker Compose

### 1. Clone the repository

```bash
git clone https://github.com/Steam-Style/ingestion.git
```

### 2. Navigate to the project directory

```bash
 cd ingestion
```

### 3. Create the shared Docker network

```bash
docker network create steam-style-shared
```

### 4. Start the services using Docker Compose

```bash
docker compose up -d
```

---

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/Steam-Style/ingestion?style=social)](https://github.com/Steam-Style/ingestion/stargazers)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Steam-Style?style=social)](https://github.com/sponsors/Steam-Style)

Made with ❤️ by the Steam Style team

[Report an Issue](https://github.com/Steam-Style/ingestion/issues) • [Visit Steam Style](https://steam.style)

</div>
