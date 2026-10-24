# Sample results

The output of the pipeline for every photo in [`sample_images/`](../../sample_images): YOLO11n detection,
CLIP ViT-B/16 embeddings, agglomerative clustering at cosine distance 0.20.

| File | Contents |
| --- | --- |
| `shelf_NN_visualization.jpg` | the photo with every product boxed in its group's colour, plus a legend |
| `shelf_NN_result.json` | the API response: boxes, confidences, group ids, groups and per-stage timings |

Regenerate with the app running: `python scripts/run_samples.py`, then `python scripts/make_readme_assets.py` to
refresh the README.
