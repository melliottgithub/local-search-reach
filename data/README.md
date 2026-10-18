# Data

There is no business data in this repo. The scans, Search Console rows and GA4 rows
belong to the businesses, so they stay on my machine.

What the scripts expect:

`geogrids.parquet`, one row per grid point of a Maps scan:
`heatmap_id, business_name, keyword, scan_date, grid_size, center_lat, center_lng,
lat, lng, distance_miles, rank, is_top3`

`gsc_daily.parquet`, one row per business (the column is called `client` in the source export),
day, device and search query:
`client, role, date, device, query, clicks, impressions, position`

`python src/prepare_data.py --source <practicum>` is how the real files are made from the
source pulls on my machine (it needs a private name map that is not in the repo).
`python src/sample_data.py` writes fake files with these columns into `data/sample/`
so everything runs. The real files go in `data/private/` (ignored by git) and you
pass the path on the command line. Both model scripts take the path as the first argument.
