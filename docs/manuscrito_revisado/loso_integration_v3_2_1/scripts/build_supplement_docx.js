// Build Supplementary_LOSO_Tables.docx from the CSVs produced by
// build_loso_reporting.py. Reads only derived CSVs already sourced from
// canonical, frozen loso_static_v1 outputs; does not hard-code scientific
// figures beyond what those CSVs contain.
// Phase LOSO_METHODS_RESULTS_INTEGRATION_V3_2_1, CP4.

const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  WidthType,
  HeadingLevel,
  AlignmentType,
} = require("docx");

const PHASE_DIR = __dirname + "/..";

function readCsv(name) {
  const raw = fs.readFileSync(path.join(PHASE_DIR, name), "utf8").trim();
  const lines = raw.split("\n");
  const header = lines[0].split(",");
  const rows = lines.slice(1).map((line) => {
    // simple CSV split that tolerates quoted commas (only used for source_sites field)
    const cells = [];
    let cur = "";
    let inQuotes = false;
    for (const ch of line) {
      if (ch === '"') {
        inQuotes = !inQuotes;
      } else if (ch === "," && !inQuotes) {
        cells.push(cur);
        cur = "";
      } else {
        cur += ch;
      }
    }
    cells.push(cur);
    const obj = {};
    header.forEach((h, i) => (obj[h] = cells[i]));
    return obj;
  });
  return rows;
}

function pct(x, digits = 1) {
  return (parseFloat(x) * 100).toFixed(digits);
}

function cellText(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 1000, type: WidthType.DXA },
    children: [
      new Paragraph({
        alignment: opts.align || AlignmentType.LEFT,
        children: [new TextRun({ text: String(text), bold: !!opts.bold, size: 20 })],
      }),
    ],
  });
}

function headerRow(labels, widths) {
  return new TableRow({
    children: labels.map((l, i) => cellText(l, { bold: true, width: widths[i] })),
  });
}

function makeTable(headers, widths, rows) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      headerRow(headers, widths),
      ...rows.map(
        (r) => new TableRow({ children: r.map((c, i) => cellText(c, { width: widths[i] })) })
      ),
    ],
  });
}

function heading(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, bold: true })] });
}

function caption(text) {
  return new Paragraph({
    spacing: { before: 120, after: 240 },
    children: [new TextRun({ text, italics: true, size: 20 })],
  });
}

function spacer() {
  return new Paragraph({ children: [new TextRun("")] });
}

// ---------------------------------------------------------------------
// S_LOSO_Design
// ---------------------------------------------------------------------
const design = readCsv("s_loso_design.csv");
const designTable = makeTable(
  ["Held-out site", "Source sites", "Held-out n (control/ADHD)", "FIT n", "Inner-validation n", "Participant/site characteristics"],
  [1600, 2200, 1800, 900, 1300, 1600],
  design.map((r) => [
    r.held_out_site,
    r.source_sites,
    `${r.held_out_n} (${r.held_out_control_n}/${r.held_out_adhd_n})`,
    r.fit_n,
    r.inner_val_n,
    r.participant_characteristics_cross_reference,
  ])
);

// ---------------------------------------------------------------------
// S_LOSO_FullMetrics
// ---------------------------------------------------------------------
const full = readCsv("s_loso_fullmetrics.csv");
const fullTable = makeTable(
  ["Held-out site", "ROI", "Model", "AUC [95% CI]", "Balanced acc.", "Macro F1", "Sensitivity", "Specificity"],
  [1300, 700, 1300, 2200, 1300, 1100, 1200, 1200],
  full.map((r) => [
    r.held_out_site,
    r.roi_set,
    r.model === "brainnetcnn" ? "BrainNetCNN" : "Logistic",
    `${pct(r.auc_point)}% [${pct(r.auc_ci_low)}, ${pct(r.auc_ci_high)}]`,
    pct(r.balanced_accuracy_point) + "%",
    pct(r.f1_macro_point) + "%",
    pct(r.sensitivity_point) + "%",
    pct(r.specificity_point) + "%",
  ])
);

// ---------------------------------------------------------------------
// S_LOSO_Contrasts (12 rows)
// ---------------------------------------------------------------------
const contrastLabel = {
  dimensionality: "116-ROI − 12-ROI (BrainNetCNN)",
  model_family_at_12: "Logistic − BrainNetCNN (12-ROI)",
  model_family_at_116: "Logistic − BrainNetCNN (116-ROI)",
};
const contrasts = readCsv("s_loso_contrasts.csv");
const contrastsTable = makeTable(
  ["Contrast", "Held-out site", "Δ AUC (pp)", "95% CI (pp)"],
  [2600, 1600, 1300, 2200],
  contrasts.map((r) => [
    contrastLabel[r.contrast] || r.contrast,
    r.held_out_site,
    pct(r.delta_point),
    `[${pct(r.delta_ci_low)}, ${pct(r.delta_ci_high)}]`,
  ])
);

// ---------------------------------------------------------------------
// S_LOSO_Seeds (BrainNetCNN only)
// ---------------------------------------------------------------------
const seeds = readCsv("s_loso_seeds.csv");
const seedsTable = makeTable(
  ["Held-out site", "ROI", "Seed AUC SD", "Seed AUC min", "Seed AUC max"],
  [1800, 900, 1600, 1600, 1600],
  seeds.map((r) => [
    r.held_out_site,
    r.roi_set,
    pct(r.seed_sd, 2) + "%",
    pct(r.seed_min) + "%",
    pct(r.seed_max) + "%",
  ])
);

// ---------------------------------------------------------------------
// S_LOSO_Convergence
// ---------------------------------------------------------------------
const conv = readCsv("s_loso_convergence.csv");
const convTable = makeTable(
  ["Held-out site", "ROI", "Runs hitting epoch 300", "Runs stopped early", "Mean epochs run", "Mean best epoch"],
  [1600, 700, 1700, 1600, 1600, 1600],
  conv.map((r) => [
    r.held_out_site,
    r.roi_set,
    `${r.n_hit_epoch_300}/${r.n_runs}`,
    r.n_stopped_before_300,
    parseFloat(r.epochs_ran_mean).toFixed(1),
    parseFloat(r.best_epoch_mean).toFixed(1),
  ])
);

const doc = new Document({
  sections: [
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({
          heading: HeadingLevel.HEADING_1,
          children: [new TextRun({ text: "Supplementary Material: Leave-One-Site-Out (LOSO) Evaluation", bold: true })],
        }),
        spacer(),

        heading("Table S_LOSO_Design. LOSO rotation design."),
        designTable,
        caption(
          "Table S_LOSO_Design. For each LOSO rotation, the held-out site, its held-out sample size (control/ADHD), the three source sites, and the FIT/inner-validation sizes pooled across those source sites. “NOT AVAILABLE IN FROZEN SCOPE” indicates that no existing, verified table of participant/site characteristics was available to cross-reference within the frozen analysis boundary; no new demographic table was created for this phase."
        ),
        spacer(),

        heading("Table S_LOSO_FullMetrics. Full metrics for all 16 held-out-site/ROI/model conditions."),
        fullTable,
        caption(
          "Table S_LOSO_FullMetrics. AUC and secondary threshold-dependent metrics (balanced accuracy, macro F1, sensitivity, specificity) at a fixed probability threshold of 0.5, for each of the 16 held-out-site × ROI × model conditions. BrainNetCNN values are means of five seed-specific runs; logistic-regression values are from a single deterministic fit. No new confidence intervals were computed for the secondary metrics."
        ),
        spacer(),

        heading("Table S_LOSO_Contrasts. Complete set of 12 preregistered LOSO contrasts."),
        contrastsTable,
        caption(
          "Table S_LOSO_Contrasts. Paired differences in AUC, in percentage points, with two-sided 95% class-stratified participant-bootstrap confidence intervals, pointwise and unadjusted for multiplicity. The same class-stratified resamples were reused across all conditions and contrasts within each held-out site, preserving pairing. No pooled or averaged estimate across sites is presented."
        ),
        spacer(),

        heading("Table S_LOSO_Seeds. BrainNetCNN between-seed dispersion."),
        seedsTable,
        caption(
          "Table S_LOSO_Seeds. Dispersion of AUC across the five BrainNetCNN initialization seeds (42–46) for each held-out site and ROI panel. This dispersion is summarized separately from the participant-bootstrap confidence intervals reported in Table 6 and in Table S_LOSO_FullMetrics, and no new inferential test was performed on it."
        ),
        spacer(),

        heading("Table S_LOSO_Convergence. Training convergence and early stopping."),
        convTable,
        caption(
          "Table S_LOSO_Convergence. Number of BrainNetCNN runs (of five per held-out site × ROI condition) that reached the 300-epoch ceiling versus stopped earlier via early stopping, with mean epochs run and mean best epoch. Summarizes the already-frozen convergence records without new analysis."
        ),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  const outPath = path.join(PHASE_DIR, "Supplementary_LOSO_Tables.docx");
  fs.writeFileSync(outPath, buffer);
  console.log("Wrote", outPath);
});
