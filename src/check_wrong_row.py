import pandas as pd

result_path = r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\output_latentspace_v2\alkis_flurstueck_synthetic_v2\20260506_141923\latent_outlier_results_v2.parquet"

res = pd.read_parquet(result_path)

target = res[res["gml_id"].astype(str).str.contains(
    "SYNTHETIC_FULLY_WRONG_ROW_0001|DENIAL2500006AK",
    na=False
)]

print(target[[
    "row_id",
    "outlier_rank",
    "gml_id",
    "combined_outlier_score",
    "umap_x",
    "umap_y",
    "cluster_label",
    "dominant_modality"
]])