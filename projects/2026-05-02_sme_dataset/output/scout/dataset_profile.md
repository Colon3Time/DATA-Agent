DATASET_PROFILE
===============
rows         : 8316
cols         : 11
dtypes       : mixed categorical/numeric
missing      : not_detected_by_csv_validation
target_column: NUMBER of MSME
problem_type : provincial_msmE_count_modeling
recommended_scaling: optional_for_linear_models
source_file  : input/sme_provincial_data.csv
dataset_type : SYNTHETIC SAMPLE - real source download blocked in current environment
selected_candidate: SME Provincial Data Thailand
selected_source: OSMEP Open Data Gateway
selected_url : https://opendata.sme.go.th/dataset/msme
download_url : https://opendata.sme.go.th/dataset/60b9b036-6a59-4291-8c4c-a56d80e4ff44/resource/8125d195-bc9a-467d-b4be-84ff661481ec/download/number-of-sme-2567.csv
license      : Creative Commons Attributions
features     : PROVINCE, REGION, SECTOR, TSIC-2 dg, TSIC-2dg_details, TSIC-5dg, TSIC-5dg_details, BUSINESS SIZE, DATA SOURCE, YEAR
target       : NUMBER of MSME
distribution : 77 Thai provinces, 2018-2023, 6 SME sectors, 3 business-size bands
synthetic_sample: yes
verdict      : Fallback synthetic sample created so the pipeline can continue; replace with real CSV when outbound HTTPS is available.
notes        : This file is generated from Scout metadata and should not be used for final statistical claims.
