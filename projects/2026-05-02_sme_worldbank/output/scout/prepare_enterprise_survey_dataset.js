const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const scoutDir = __dirname;
const projectDir = path.resolve(scoutDir, "..", "..");
const inputDir = path.join(projectDir, "input");
const logDir = path.join(projectDir, "logs");
const pySource = path.join(scoutDir, "prepare_enterprise_survey_dataset.py");

const attempts = [
  ["world_bank_indicator_api", "https://api.worldbank.org/v2/country/TH/indicator/IC.BUS.EASE.XQ?format=json", "json"],
  ["world_bank_enterprise_survey_databank_zip", "https://databank.worldbank.org/data/download/Enterprise-Survey-Data.zip", "zip"],
  ["kaggle_world_bank_enterprise_surveys", "https://www.kaggle.com/datasets/worldbank/world-bank-enterprise-surveys", "html"],
  ["world_bank_microdata_enterprise_surveys_catalog", "https://microdata.worldbank.org/index.php/catalog/enterprise_surveys", "html"],
];

function ensureDirs() {
  for (const dir of [inputDir, scoutDir, logDir]) fs.mkdirSync(dir, { recursive: true });
}

function rel(filePath) {
  return path.relative(projectDir, filePath).replaceAll(path.sep, "/");
}

function logLine(message) {
  const stamp = new Date().toISOString().replace("T", " ").slice(0, 19);
  fs.appendFileSync(path.join(logDir, "dataset_preparation.log"), `[${stamp}] ${message}\n`, "utf8");
  console.log(message);
}

function stableFraction(...parts) {
  const hash = crypto.createHash("sha256").update(parts.join("|")).digest("hex").slice(0, 12);
  return parseInt(hash, 16) / (16 ** 12 - 1);
}

function bounded(value, lo, hi, digits = 2) {
  const mult = 10 ** digits;
  return Math.round(Math.max(lo, Math.min(hi, value)) * mult) / mult;
}

function parseProvinces() {
  const source = fs.readFileSync(pySource, "utf8");
  const rows = [];
  const pattern = /\("([^"]+)",\s*"([^"]+)",\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\),/g;
  let match;
  while ((match = pattern.exec(source))) {
    rows.push({
      province: match[1],
      region: match[2],
      gppPc: Number(match[3]),
      populationM: Number(match[4]),
      agri: Number(match[5]),
      industry: Number(match[6]),
      services: Number(match[7]),
      urban: Number(match[8]),
    });
  }
  if (rows.length !== 77) throw new Error(`Expected 77 provinces, parsed ${rows.length}`);
  return rows;
}

async function tryDownloads() {
  const results = [];
  for (const [name, url, kind] of attempts) {
    const ext = kind === "json" ? ".json" : kind === "zip" ? ".zip" : ".html";
    const target = path.join(inputDir, `${name}${ext}`);
    const started = Date.now();
    const result = { name, url, target: rel(target), status: "started" };
    logLine(`DOWNLOAD_ATTEMPT start name=${name} url=${url}`);
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 25000);
      const response = await fetch(url, {
        signal: controller.signal,
        headers: {
          "user-agent": "Mozilla/5.0 DATA-Agent dataset preparation",
          accept: "application/json,text/csv,application/zip,text/html,*/*",
        },
      });
      clearTimeout(timer);
      const buffer = Buffer.from(await response.arrayBuffer());
      fs.writeFileSync(target, buffer);
      result.status = response.ok ? "downloaded" : "http_error";
      result.http_status = response.status;
      result.content_type = response.headers.get("content-type") || "";
      result.bytes = buffer.length;
      result.elapsed_seconds = Math.round((Date.now() - started) / 10) / 100;
      if (kind === "zip") {
        const zipMagic = buffer.slice(0, 2).toString("utf8");
        if (zipMagic !== "PK") result.status = "downloaded_but_invalid_zip";
      }
      logLine(`DOWNLOAD_ATTEMPT result name=${name} status=${result.status} bytes=${result.bytes} content_type=${result.content_type}`);
    } catch (error) {
      result.status = "failed";
      result.error = `${error.name || "Error"}: ${error.message}`;
      result.elapsed_seconds = Math.round((Date.now() - started) / 10) / 100;
      logLine(`DOWNLOAD_ATTEMPT failed name=${name} status=failed error=${result.error}`);
    }
    results.push(result);
  }
  return results;
}

function dominantSector(p) {
  if (p.industry >= p.agri && p.industry >= p.services) return "manufacturing";
  if (p.services >= p.agri) return "services";
  return "agro_processing";
}

function makeRows(provinces) {
  const industryList = [
    ["food_products", "manufacturing"],
    ["garments", "manufacturing"],
    ["chemicals_plastics", "manufacturing"],
    ["fabricated_metals", "manufacturing"],
    ["retail_trade", "services"],
    ["hotel_restaurant", "services"],
    ["transport_logistics", "services"],
    ["business_services", "services"],
    ["agro_processing", "agro_processing"],
    ["construction_materials", "manufacturing"],
  ];
  const legal = ["sole_proprietorship", "partnership", "limited_company"];
  const rows = [];
  provinces.forEach((p, provinceIndex0) => {
    const provinceIndex = provinceIndex0 + 1;
    const scale = Math.log10(Math.max(p.gppPc, 1));
    const sector = dominantSector(p);
    for (let firmNo = 1; firmNo <= 10; firmNo += 1) {
      const f = stableFraction(p.province, firmNo);
      let sizeClass = "small";
      let employees = 8 + Math.floor(18 * f + 10 * p.urban + 6 * p.industry);
      if ([8, 9].includes(firmNo)) {
        sizeClass = "medium";
        employees = 35 + Math.floor(65 * stableFraction(p.province, firmNo, "m"));
      }
      if (firmNo === 10 && p.gppPc > 180000) {
        sizeClass = "large";
        employees = 110 + Math.floor(260 * stableFraction(p.province, "large"));
      }
      let pool = industryList.filter((item) => item[1] === sector);
      if (!pool.length) pool = industryList;
      let [industryName, industryGroup] = pool[firmNo % pool.length];
      if (firmNo % 3 === 0) [industryName, industryGroup] = industryList[(provinceIndex + firmNo) % industryList.length];
      const salesM = (employees * (0.42 + scale * 0.16) * (0.75 + stableFraction(p.province, firmNo, "sales"))) / 10;
      const exporter = Number((p.industry > 0.45 || ["Chonburi", "Rayong", "Samut Prakan", "Samut Sakhon"].includes(p.province)) && [7, 8, 9, 10].includes(firmNo));
      const foreignOwned = bounded((0.01 + 0.08 * p.urban + 0.1 * p.industry + 0.05 * exporter) * 100, 0, 35, 1);
      const powerOutages = bounded(10 - 5 * p.urban + 4 * p.agri + 2 * stableFraction(p.province, firmNo, "power"), 0, 18, 1);
      const loan = Number(stableFraction(p.province, firmNo, "loan") < 0.34 + 0.18 * p.urban + 0.08 * p.industry);
      const obstacleBase = (1 - p.urban) + p.agri * 0.7;
      rows.push({
        survey_id: "THA_SIM_ES_2026",
        source_status: "Simulated from public World Bank metadata",
        country_code: "THA",
        country_name: "Thailand",
        survey_year: 2026,
        wb_region: "East Asia & Pacific",
        province_code: `TH-${String(provinceIndex).padStart(2, "0")}`,
        province_name: p.province,
        thai_region: p.region,
        firm_id: `THA-${String(provinceIndex).padStart(2, "0")}-${String(firmNo).padStart(3, "0")}`,
        strata_region: p.region,
        strata_size: sizeClass,
        strata_sector: industryGroup,
        industry_isic_like: industryName,
        legal_status: legal[(provinceIndex + firmNo) % legal.length],
        years_operating: 2 + Math.floor(28 * stableFraction(p.province, firmNo, "age")),
        permanent_full_time_workers: employees,
        temporary_workers: Math.floor(employees * bounded(0.04 + 0.13 * p.agri + 0.08 * stableFraction(p.province, firmNo, "temp"), 0, 0.35, 3)),
        female_full_time_workers_pct: bounded(24 + 22 * p.services + 8 * stableFraction(p.province, firmNo, "fempct"), 12, 70, 1),
        female_top_manager: Number(stableFraction(p.province, firmNo, "ftm") < 0.18 + 0.12 * p.services),
        female_ownership: Number(stableFraction(p.province, firmNo, "female") > 0.58),
        foreign_ownership_pct: foreignOwned,
        private_domestic_ownership_pct: bounded(100 - foreignOwned, 65, 100, 1),
        state_ownership_pct: 0,
        exporter_direct: exporter,
        direct_exports_pct_sales: bounded(exporter * (8 + 30 * stableFraction(p.province, firmNo, "exportpct")), 0, 60, 1),
        indirect_exports_pct_sales: bounded((exporter || p.industry > 0.45 ? 1 : 0) * (4 + 16 * stableFraction(p.province, firmNo, "iexport")), 0, 35, 1),
        annual_sales_million_thb: bounded(salesM, 0.4, 650, 2),
        sales_growth_pct: bounded(-4 + 8 * p.urban + 3 * p.services + 8 * stableFraction(p.province, firmNo, "growth"), -12, 24, 1),
        capacity_utilization_pct: bounded(55 + 18 * p.urban + 8 * p.industry + 10 * stableFraction(p.province, firmNo, "cap"), 45, 96, 1),
        labor_productivity_thb_per_worker: Math.floor((salesM * 1000000) / Math.max(employees, 1)),
        uses_email: Number(stableFraction(p.province, firmNo, "email") < 0.62 + 0.28 * p.urban),
        has_website: Number(stableFraction(p.province, firmNo, "web") < 0.38 + 0.36 * p.urban + 0.12 * p.services),
        uses_digital_payments: Number(stableFraction(p.province, firmNo, "pay") < 0.5 + 0.3 * p.urban + 0.1 * p.services),
        introduced_new_product: Number(stableFraction(p.province, firmNo, "innov") < 0.22 + 0.18 * p.industry + 0.12 * p.urban),
        has_international_quality_certification: Number(stableFraction(p.province, firmNo, "cert") < 0.08 + 0.18 * p.industry + 0.1 * exporter),
        uses_foreign_licensed_technology: Number(stableFraction(p.province, firmNo, "tech") < 0.06 + 0.14 * p.industry + 0.08 * foreignOwned / 100),
        formal_training: Number(stableFraction(p.province, firmNo, "training") < 0.18 + 0.16 * p.urban + 0.1 * (sizeClass !== "small")),
        power_outages_count_year: powerOutages,
        losses_due_power_outages_pct_sales: bounded(powerOutages * 0.18 + stableFraction(p.province, firmNo, "loss"), 0, 6, 2),
        generator_owned: Number(stableFraction(p.province, firmNo, "generator") < 0.12 + 0.1 * powerOutages / 10),
        water_insufficiencies_count_year: bounded(3 + 7 * p.agri + 2 * stableFraction(p.province, firmNo, "water"), 0, 16, 1),
        days_to_get_electricity: Math.round(bounded(12 + 14 * (1 - p.urban) + 8 * stableFraction(p.province, firmNo, "elecday"), 5, 55, 0)),
        days_to_get_construction_permit: Math.round(bounded(25 + 18 * p.urban + 20 * stableFraction(p.province, firmNo, "permit"), 12, 95, 0)),
        days_to_clear_exports_customs: Math.round(bounded(2 + 5 * exporter + 5 * stableFraction(p.province, firmNo, "customs"), 1, 18, 0)),
        senior_management_time_regulation_pct: bounded(5 + 5 * p.urban + 5 * stableFraction(p.province, firmNo, "regtime"), 1, 22, 1),
        informal_payments_expected: Number(stableFraction(p.province, firmNo, "gift") < 0.04 + 0.06 * obstacleBase),
        security_costs_pct_sales: bounded(0.3 + 0.9 * p.urban + 0.8 * stableFraction(p.province, firmNo, "security"), 0, 4, 2),
        losses_due_theft_pct_sales: bounded(0.1 + 0.8 * stableFraction(p.province, firmNo, "theft") + 0.3 * (1 - p.urban), 0, 4, 2),
        checking_savings_account: Number(stableFraction(p.province, firmNo, "acct") < 0.72 + 0.2 * p.urban),
        line_of_credit_or_loan: loan,
        loan_rejected_recently: Number(!loan && stableFraction(p.province, firmNo, "reject") < 0.16 + 0.12 * (1 - p.urban)),
        working_capital_financed_by_banks_pct: bounded(8 + 20 * loan + 12 * p.urban + 10 * stableFraction(p.province, firmNo, "wcbank"), 0, 70, 1),
        working_capital_financed_by_supplier_credit_pct: bounded(5 + 13 * stableFraction(p.province, firmNo, "supplier"), 0, 35, 1),
        investment_financed_by_banks_pct: bounded(5 + 25 * loan + 10 * p.industry + 10 * stableFraction(p.province, firmNo, "invbank"), 0, 80, 1),
        collateral_required: Number(loan && stableFraction(p.province, firmNo, "collateralreq") < 0.7),
        collateral_value_pct_loan: loan ? bounded(130 + 28 * (1 - p.urban) + 20 * stableFraction(p.province, firmNo, "collateral"), 80, 220, 1) : "",
        tax_rate_obstacle: Number(stableFraction(p.province, firmNo, "tax") < 0.28 + 0.08 * obstacleBase),
        tax_administration_obstacle: Number(stableFraction(p.province, firmNo, "taxadmin") < 0.22 + 0.08 * obstacleBase),
        customs_trade_regulations_obstacle: Number(stableFraction(p.province, firmNo, "tradeobs") < 0.14 + 0.16 * exporter),
        labor_regulations_obstacle: Number(stableFraction(p.province, firmNo, "laborobs") < 0.18 + 0.08 * p.urban),
        inadequately_educated_workforce_obstacle: Number(stableFraction(p.province, firmNo, "skillobs") < 0.2 + 0.1 * p.industry + 0.08 * (1 - p.urban)),
        access_to_finance_obstacle: Number(stableFraction(p.province, firmNo, "finobs") < 0.24 + 0.16 * (1 - p.urban) - 0.08 * loan),
        electricity_obstacle: Number(stableFraction(p.province, firmNo, "elecobs") < 0.16 + 0.08 * powerOutages / 10),
        transport_obstacle: Number(stableFraction(p.province, firmNo, "transportobs") < 0.14 + 0.12 * (1 - p.urban)),
        political_instability_obstacle: Number(stableFraction(p.province, firmNo, "polobs") < 0.18),
        corruption_obstacle: Number(stableFraction(p.province, firmNo, "corrobs") < 0.16 + 0.05 * obstacleBase),
        informal_competition_obstacle: Number(stableFraction(p.province, firmNo, "informalobs") < bounded(45 - 18 * p.urban + 10 * p.agri + 5 * stableFraction(p.province, firmNo, "informal"), 8, 65, 1) / 100),
        practices_of_informal_competitors_pct: bounded(45 - 18 * p.urban + 10 * p.agri + 5 * stableFraction(p.province, firmNo, "informal"), 8, 65, 1),
        court_system_obstacle: Number(stableFraction(p.province, firmNo, "courtobs") < 0.12 + 0.05 * obstacleBase),
        crime_theft_disorder_obstacle: Number(stableFraction(p.province, firmNo, "crimeobs") < 0.1 + 0.04 * p.urban),
        province_gpp_per_capita_proxy_thb: p.gppPc,
        province_population_proxy_million: p.populationM,
        province_agriculture_share_proxy: p.agri,
        province_industry_share_proxy: p.industry,
        province_services_share_proxy: p.services,
        province_urbanization_proxy: p.urban,
        weight: bounded((p.populationM / 0.75) * (0.7 + 0.6 * stableFraction(p.province, firmNo, "weight")), 0.08, 12, 4),
      });
    }
  });
  return rows;
}

function csvValue(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function writeDataset(rows) {
  const target = path.join(inputDir, "thailand_enterprise_surveys_simulated_2026.csv");
  const columns = Object.keys(rows[0]);
  const csv = [columns.join(","), ...rows.map((row) => columns.map((col) => csvValue(row[col])).join(","))].join("\n") + "\n";
  fs.writeFileSync(target, csv, "utf8");
  logLine(`FALLBACK_DATASET created path=${rel(target)} rows=${rows.length} columns=${columns.length}`);
  return target;
}

function writeManifest(attemptResults, datasetPath, rows) {
  const manifestPath = path.join(inputDir, "dataset_manifest.json");
  const manifest = {
    created_at: new Date().toISOString(),
    project: path.basename(projectDir),
    attempts: attemptResults,
    final_dataset: rel(datasetPath),
    final_dataset_status: "Simulated from public World Bank metadata",
    rows: rows.length,
    columns: Object.keys(rows[0]).length,
    unique_provinces: new Set(rows.map((row) => row.province_name)).size,
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf8");
  logLine(`MANIFEST written path=${rel(manifestPath)}`);
}

function writeProfile(attemptResults, datasetPath, rows) {
  const profilePath = path.join(scoutDir, "dataset_profile.md");
  const columns = Object.keys(rows[0]);
  let profile = `# Dataset Profile: Thailand Enterprise Surveys SME Dataset

## STATUS
- Dataset file: \`${rel(datasetPath)}\`
- Dataset status: **Simulated from public World Bank metadata**
- Rows: ${rows.length}
- Columns: ${columns.length}
- Province coverage: 77 Thai provinces, 10 modeled firm records per province
- Pipeline readiness: ready for ingestion from \`input/\`

## Why this file exists
The workflow attempted every requested network source first. The workspace network path did not provide a directly usable World Bank Enterprise Surveys microdata CSV/ZIP for Thailand, so the pipeline now uses a file-based fallback. This fallback is an analysis-ready CSV with Enterprise Surveys-style fields and deterministic values calibrated from public metadata concepts: World Bank Enterprise Surveys questionnaire themes, Thailand province names, broad regional structure, and province-level economic/population proxy tiers.

This is **not official respondent-level World Bank microdata**. Treat it as a realistic scaffold for pipeline development, schema validation, feature engineering, and dashboard testing until licensed/official microdata is available.

## Download attempts
| Source | URL | Result | Local artifact |
| --- | --- | --- | --- |
`;
  for (const item of attemptResults) {
    profile += `| ${item.name} | ${item.url} | ${item.status}${item.http_status ? ` HTTP ${item.http_status}` : ""} | \`${item.target}\` |\n`;
  }
  profile += `
## Input file contract
- Encoding: UTF-8
- Format: CSV with header
- Primary key: \`firm_id\`
- Geography keys: \`province_code\`, \`province_name\`, \`thai_region\`, \`country_code\`
- Survey keys: \`survey_id\`, \`survey_year\`, \`strata_region\`, \`strata_size\`, \`strata_sector\`, \`weight\`
- Source marker: every row contains \`source_status = "Simulated from public World Bank metadata"\`

## Column groups
- Identification, stratification, firm demographics, performance, digital adoption, innovation, infrastructure, regulation, finance, obstacles, and province proxy fields.

## DATASET_RISK_REGISTER
| Risk | Severity | Mitigation |
| --- | --- | --- |
| Simulated data may be mistaken for official survey microdata | High | \`source_status\` column and this profile state the status clearly. Do not publish as official World Bank respondent data. |
| Province-level economic proxy values are approximate tiers, not audited official series | Medium | Replace proxy fields when official NESDC/DOPA tables are available locally. |
| Modeled firm outcomes are deterministic and useful for pipeline testing, not statistical inference | High | Use for ETL, validation, and interface development only. |
| Kaggle source may require authenticated CLI/API access | Medium | Log records the attempt; retry later with Kaggle credentials if needed. |
| Microdata catalog may require manual license/terms flow | Medium | Use official World Bank Microdata Library process before analytical publication. |

## Validation summary
- Exactly 77 unique provinces: ${new Set(rows.map((row) => row.province_name)).size}
- Rows per province: 10
- More than 50 columns: ${columns.length} columns
- Missing \`firm_id\`: ${rows.filter((row) => !row.firm_id).length}
- Downloaded non-fallback artifacts: ${attemptResults.filter((item) => item.status === "downloaded").length}

## Next pipeline step
Read \`input/thailand_enterprise_surveys_simulated_2026.csv\`, validate \`source_status\`, and keep this profile attached to downstream outputs so simulated provenance is not lost.
`;
  fs.writeFileSync(profilePath, profile, "utf8");
  logLine(`PROFILE written path=${rel(profilePath)}`);
}

async function main() {
  ensureDirs();
  logLine("DATASET_PREPARATION start");
  const attemptResults = await tryDownloads();
  const provinces = parseProvinces();
  const rows = makeRows(provinces);
  const datasetPath = writeDataset(rows);
  writeManifest(attemptResults, datasetPath, rows);
  writeProfile(attemptResults, datasetPath, rows);
  logLine("DATASET_PREPARATION complete");
}

main().catch((error) => {
  logLine(`DATASET_PREPARATION failed error=${error.stack || error.message}`);
  process.exitCode = 1;
});
