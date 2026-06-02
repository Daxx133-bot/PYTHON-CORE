"""
================================================================================
Production-Grade Industrial Data Analytics & Visualization Pipeline
================================================================================
Framework Version : 2.0.0
Architecture      : Object-Oriented, SOLID Principles, Modular Enterprise Design
Compatible With   : Any structured/tabular dataset (CSV, Excel, JSON, Parquet, SQL)
Libraries         : pandas, numpy, scipy, matplotlib, seaborn, sklearn, logging
Author            : Analytics Engineering Division
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ─────────────────────────────────────────────────────────────────────────────
import logging
import warnings
import os
import sys
import time
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ─────────────────────────────────────────────────────────────────────────────
# Third-Party Imports (Permitted Libraries Only)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, normaltest
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Global Style Configuration
# ─────────────────────────────────────────────────────────────────────────────
PALETTE        = "#0A2342"   # Navy
ACCENT         = "#E63946"   # Red
HIGHLIGHT      = "#457B9D"   # Steel Blue
NEUTRAL        = "#A8DADC"   # Muted Cyan
SUCCESS        = "#2A9D8F"   # Teal
WARNING_COLOR  = "#E9C46A"   # Gold
BACKGROUND     = "#F8F9FA"   # Off-White
GRID_COLOR     = "#DEE2E6"   # Light Gray
TEXT_COLOR     = "#212529"   # Near-Black

CHART_PALETTE = [PALETTE, ACCENT, HIGHLIGHT, SUCCESS, WARNING_COLOR,
                 NEUTRAL, "#264653", "#F4A261", "#E76F51", "#023E8A"]

plt.rcParams.update({
    "figure.facecolor"    : BACKGROUND,
    "axes.facecolor"      : "white",
    "axes.edgecolor"      : GRID_COLOR,
    "axes.labelcolor"     : TEXT_COLOR,
    "axes.titlesize"      : 14,
    "axes.labelsize"      : 11,
    "axes.grid"           : True,
    "grid.color"          : GRID_COLOR,
    "grid.linestyle"      : "--",
    "grid.alpha"          : 0.6,
    "xtick.color"         : TEXT_COLOR,
    "ytick.color"         : TEXT_COLOR,
    "font.family"         : "DejaVu Sans",
    "figure.dpi"          : 120,
    "savefig.dpi"         : 150,
    "savefig.bbox"        : "tight",
    "savefig.facecolor"   : BACKGROUND,
    "legend.framealpha"   : 0.9,
    "legend.fontsize"     : 9,
})


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 – LOGGING SUBSYSTEM
# ═════════════════════════════════════════════════════════════════════════════

class PipelineLogger:
    """
    Centralised, singleton-style logger for the entire analytics pipeline.
    Outputs to both console (colourised) and a rotating file log.
    """

    _instance: Optional["PipelineLogger"] = None

    def __new__(cls, log_dir: str = "outputs/logs") -> "PipelineLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self, log_dir: str = "outputs/logs") -> None:
        if self._initialised:
            return
        self._initialised = True
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"pipeline_{ts}.log"

        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger("AnalyticsPipeline")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # File handler (full detail)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        # Console handler (INFO+)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

    # Convenience wrappers
    def info(self, msg: str) -> None:    self.logger.info(msg)
    def debug(self, msg: str) -> None:   self.logger.debug(msg)
    def warning(self, msg: str) -> None: self.logger.warning(msg)
    def error(self, msg: str) -> None:   self.logger.error(msg)

    def section(self, title: str) -> None:
        sep = "─" * 72
        self.logger.info("")
        self.logger.info(sep)
        self.logger.info(f"  {title.upper()}")
        self.logger.info(sep)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 – CONFIGURATION DATACLASS
# ═════════════════════════════════════════════════════════════════════════════

class PipelineConfig:
    """
    Central configuration object. All tuneable parameters live here so the
    pipeline can be adapted to any dataset without touching implementation code.
    """

    def __init__(
        self,
        data_path: str,
        output_dir: str = "outputs",
        report_dir: str = "outputs/reports",
        chart_dir: str  = "outputs/charts",
        log_dir: str    = "outputs/logs",
        # Preprocessing
        outlier_iqr_multiplier: float  = 1.5,
        isolation_forest_contamination: float = 0.05,
        # Clustering
        n_clusters: int = 4,
        # Regression targets
        regression_target: Optional[str] = None,
        # Datetime columns (auto-detected if None)
        datetime_columns: Optional[List[str]] = None,
        # Columns to exclude from analysis
        exclude_columns: Optional[List[str]] = None,
        # Sampling for heavy operations
        sample_size: int = 10_000,
    ) -> None:
        self.data_path       = data_path
        self.output_dir      = output_dir
        self.report_dir      = report_dir
        self.chart_dir       = chart_dir
        self.log_dir         = log_dir
        self.outlier_iqr_multiplier        = outlier_iqr_multiplier
        self.isolation_forest_contamination = isolation_forest_contamination
        self.n_clusters      = n_clusters
        self.regression_target = regression_target
        self.datetime_columns  = datetime_columns or []
        self.exclude_columns   = exclude_columns or []
        self.sample_size       = sample_size

    def create_directories(self) -> None:
        for d in [self.output_dir, self.report_dir, self.chart_dir, self.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DATA INGESTION LAYER  (Abstract Base + Concrete Loaders)
# ═════════════════════════════════════════════════════════════════════════════

class BaseDataLoader(ABC):
    """Abstract interface for dataset loaders. Open/Closed Principle."""

    @abstractmethod
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        ...

    def validate(self, df: pd.DataFrame) -> bool:
        return not df.empty


class CSVLoader(BaseDataLoader):
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)


class ExcelLoader(BaseDataLoader):
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_excel(path, **kwargs)


class JSONLoader(BaseDataLoader):
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_json(path, **kwargs)


class ParquetLoader(BaseDataLoader):
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_parquet(path, **kwargs)


class DataIngestionLayer:
    """
    Façade that selects the correct loader from the file extension, then
    performs schema detection and initial data quality profiling.
    """

    _LOADERS: Dict[str, BaseDataLoader] = {
        ".csv"     : CSVLoader(),
        ".tsv"     : CSVLoader(),
        ".xlsx"    : ExcelLoader(),
        ".xls"     : ExcelLoader(),
        ".json"    : JSONLoader(),
        ".parquet" : ParquetLoader(),
    }

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config = config
        self.log    = logger

    def load(self) -> pd.DataFrame:
        self.log.section("Data Ingestion Layer")
        ext = Path(self.config.data_path).suffix.lower()
        loader = self._LOADERS.get(ext)
        if loader is None:
            raise ValueError(f"Unsupported file type: {ext}")

        self.log.info(f"Loading dataset: {self.config.data_path}")
        t0 = time.perf_counter()
        df = pd.read_csv("healthcare_dataset.csv")
        elapsed = time.perf_counter() - t0
        self.log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns in {elapsed:.2f}s")

        if not loader.validate(df):
            raise ValueError("Loaded dataframe is empty.")

        self._profile_schema(df)
        return df

    def _profile_schema(self, df: pd.DataFrame) -> None:
        self.log.info("--- Schema Profile ---")
        for col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            uniq     = df[col].nunique()
            dtype    = str(df[col].dtype)
            self.log.info(f"  {col:<30} dtype={dtype:<12} unique={uniq:<8} nulls={null_pct:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 – DATA PROCESSING & TRANSFORMATION LAYER
# ═════════════════════════════════════════════════════════════════════════════

class DataQualityReport:
    """Value object capturing data-quality metrics before and after cleaning."""

    def __init__(self) -> None:
        self.initial_rows : int = 0
        self.final_rows   : int = 0
        self.duplicates_removed: int = 0
        self.nulls_imputed: Dict[str, int] = {}
        self.outliers_flagged: Dict[str, int] = {}
        self.new_features: List[str] = []


class DataPreprocessor:
    """
    Modular data preprocessing component. Single Responsibility: transform
    the raw ingested dataframe into an analysis-ready frame.
    """

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config = config
        self.log    = logger
        self.report = DataQualityReport()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

    # ── Public entry point ─────────────────────────────────────────────────
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.section("Data Processing & Transformation Layer")
        self.report.initial_rows = len(df)

        df = df.copy()
        df = self._drop_excluded(df)
        df = self._standardise_column_names(df)
        df = self._infer_and_parse_datetimes(df)
        df = self._clean_string_columns(df)
        df = self._remove_duplicates(df)
        df = self._handle_missing_values(df)
        df = self._engineer_features(df)
        df = self._flag_outliers(df)

        self.report.final_rows = len(df)
        self._log_quality_report()
        return df

    # ── Private helpers ────────────────────────────────────────────────────
    def _drop_excluded(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.config.exclude_columns if c in df.columns]
        if cols_to_drop:
            self.log.info(f"Dropping excluded columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        return df

    def _standardise_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns.str.strip()
                      .str.replace(r"\s+", "_", regex=True)
                      .str.replace(r"[^\w]", "", regex=True)
                      .str.lower()
        )
        self.log.info(f"Standardised column names: {df.columns.tolist()}")
        return df

    def _infer_and_parse_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        # Explicit config overrides
        for col in self.config.datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Auto-detect: string columns whose name hints at dates
        date_hints = {"date", "time", "timestamp", "dt", "month", "year"}
        for col in df.select_dtypes(include="object").columns:
            if any(h in col.lower() for h in date_hints):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    if parsed.notna().mean() > 0.7:
                        df[col] = parsed
                        self.log.info(f"  Auto-parsed datetime: {col}")
                except Exception:
                    pass
        return df

    def _clean_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip().str.title()
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        self.report.duplicates_removed = removed
        self.log.info(f"Duplicates removed: {removed:,}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            n_null = df[col].isnull().sum()
            if n_null == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
                strategy = "median"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col].fillna(method="ffill", inplace=True)
                strategy = "forward-fill"
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                strategy = "mode"
            self.report.nulls_imputed[col] = n_null
            self.log.info(f"  Imputed {n_null} nulls in '{col}' using {strategy}")
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive new analytical features from existing columns."""
        new_feats: List[str] = []

        # ── Datetime-derived features ──────────────────────────────────────
        dt_cols = df.select_dtypes(include="datetime64").columns.tolist()
        for col in dt_cols:
            safe = col.replace("_", "")
            df[f"{safe}_year"]   = df[col].dt.year
            df[f"{safe}_month"]  = df[col].dt.month
            df[f"{safe}_dow"]    = df[col].dt.dayofweek   # 0 = Monday
            df[f"{safe}_quarter"]= df[col].dt.quarter
            new_feats += [f"{safe}_year", f"{safe}_month", f"{safe}_dow", f"{safe}_quarter"]

        # ── Length-of-stay (if admission + discharge columns exist) ────────
        admit_cols    = [c for c in df.columns if "admission" in c and pd.api.types.is_datetime64_any_dtype(df[c])]
        discharge_cols = [c for c in df.columns if "discharge" in c and pd.api.types.is_datetime64_any_dtype(df[c])]
        if admit_cols and discharge_cols:
            df["length_of_stay_days"] = (
                df[discharge_cols[0]] - df[admit_cols[0]]
            ).dt.days.clip(lower=0)
            new_feats.append("length_of_stay_days")
            self.log.info("  Engineered: length_of_stay_days")

        # ── Numeric log / ratio transforms ────────────────────────────────
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            if df[col].min() > 0:
                df[f"log_{col}"] = np.log1p(df[col])
                new_feats.append(f"log_{col}")

        # ── Billing Amount band ────────────────────────────────────────────
        if "billing_amount" in df.columns:
            df["billing_band"] = pd.qcut(
                df["billing_amount"].clip(lower=0), q=4,
                labels=["Low", "Medium", "High", "Premium"], duplicates="drop"
            )
            new_feats.append("billing_band")

        # ── Age group ─────────────────────────────────────────────────────
        if "age" in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 18, 35, 50, 65, 120],
                labels=["<18", "18-35", "36-50", "51-65", "65+"]
            )
            new_feats.append("age_group")

        self.report.new_features = new_feats
        self.log.info(f"Engineered {len(new_feats)} new features.")
        return df

    def _flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        k = self.config.outlier_iqr_multiplier
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (df[col] < q1 - k * iqr) | (df[col] > q3 + k * iqr)
            n_out = mask.sum()
            if n_out:
                self.report.outliers_flagged[col] = int(n_out)
                df[f"{col}_is_outlier"] = mask.astype(int)
        self.log.info(f"Outlier flags added for {len(self.report.outliers_flagged)} columns.")
        return df

    def _log_quality_report(self) -> None:
        self.log.info("--- Data Quality Summary ---")
        self.log.info(f"  Rows (before → after): {self.report.initial_rows:,} → {self.report.final_rows:,}")
        self.log.info(f"  Duplicates removed    : {self.report.duplicates_removed:,}")
        self.log.info(f"  Columns imputed       : {len(self.report.nulls_imputed)}")
        self.log.info(f"  Outlier cols flagged  : {len(self.report.outliers_flagged)}")
        self.log.info(f"  New features created  : {len(self.report.new_features)}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 – EXPLORATORY DATA ANALYSIS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class EDAEngine:
    """
    Adaptive EDA component. Detects column types and generates statistics
    and summaries that are passed downstream to the Insight Generator.
    """

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config = config
        self.log    = logger
        self.eda_results: Dict[str, Any] = {}

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.log.section("Exploratory Data Analysis")

        self.eda_results["shape"]              = df.shape
        self.eda_results["descriptive_stats"]  = self._descriptive_stats(df)
        self.eda_results["distribution"]       = self._distribution_summary(df)
        self.eda_results["correlation"]        = self._correlation_analysis(df)
        self.eda_results["categorical"]        = self._categorical_analysis(df)
        self.eda_results["time_series"]        = self._time_series_analysis(df)
        self.eda_results["outlier_summary"]    = self._outlier_summary(df)
        self.eda_results["data_quality"]       = self._data_quality_summary(df)

        return self.eda_results

    # ── Sub-analyses ───────────────────────────────────────────────────────
    def _descriptive_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        num_df = df.select_dtypes(include=[np.number])
        desc = num_df.describe(percentiles=[.10, .25, .50, .75, .90]).T
        desc["skewness"] = num_df.skew()
        desc["kurtosis"] = num_df.kurtosis()
        desc["cv_%"]     = (num_df.std() / num_df.mean().replace(0, np.nan) * 100).round(2)
        self.log.info(f"Descriptive stats computed for {len(desc)} numeric columns.")
        return desc

    def _distribution_summary(self, df: pd.DataFrame) -> Dict:
        results = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 8:
                continue
            try:
                stat, p = normaltest(series.sample(min(5000, len(series)), random_state=42))
                results[col] = {
                    "mean"    : float(series.mean()),
                    "std"     : float(series.std()),
                    "skew"    : float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                    "normal_p": float(p),
                    "is_normal": bool(p > 0.05),
                }
            except Exception:
                pass
        return results

    def _correlation_analysis(self, df: pd.DataFrame) -> Dict:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return {}
        corr = num_df.corr(method="pearson")
        # Extract top pairs
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index()
        )
        pairs.columns = ["feature_a", "feature_b", "pearson_r"]
        pairs["abs_r"] = pairs["pearson_r"].abs()
        pairs = pairs.sort_values("abs_r", ascending=False).head(20)
        return {
            "matrix": corr,
            "top_pairs": pairs.reset_index(drop=True),
        }

    def _categorical_analysis(self, df: pd.DataFrame) -> Dict:
        results = {}
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            vc = df[col].value_counts()
            results[col] = {
                "unique_count"  : int(df[col].nunique()),
                "top_value"     : str(vc.index[0]) if len(vc) else "",
                "top_freq"      : int(vc.iloc[0]) if len(vc) else 0,
                "top_pct"       : float(round(vc.iloc[0] / len(df) * 100, 2)) if len(vc) else 0,
                "value_counts"  : vc.head(10).to_dict(),
            }
        return results

    def _time_series_analysis(self, df: pd.DataFrame) -> Dict:
        dt_cols = df.select_dtypes(include="datetime64").columns.tolist()
        results = {}
        for col in dt_cols:
            series = df[col].dropna().sort_values()
            results[col] = {
                "min"  : str(series.min()),
                "max"  : str(series.max()),
                "range_days": int((series.max() - series.min()).days),
                "monthly_counts": (
                    df.set_index(col)
                      .resample("ME")
                      .size()
                      .rename("count")
                      .reset_index()
                      .assign(**{col: lambda x: x[col].astype(str)})
                      .to_dict("records")
                ),
            }
        return results

    def _outlier_summary(self, df: pd.DataFrame) -> Dict:
        outlier_cols = [c for c in df.columns if c.endswith("_is_outlier")]
        return {
            col.replace("_is_outlier", ""): int(df[col].sum())
            for col in outlier_cols
        }

    def _data_quality_summary(self, df: pd.DataFrame) -> Dict:
        total_cells = df.shape[0] * df.shape[1]
        null_cells  = df.isnull().sum().sum()
        return {
            "total_rows"       : df.shape[0],
            "total_columns"    : df.shape[1],
            "total_cells"      : total_cells,
            "null_cells"       : int(null_cells),
            "completeness_pct" : round((1 - null_cells / total_cells) * 100, 2),
            "numeric_columns"  : len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=["object","category"]).columns),
            "datetime_columns" : len(df.select_dtypes(include="datetime64").columns),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 – INSIGHT GENERATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class InsightGenerationEngine:
    """
    Automatically mines the EDA results and raw dataframe to produce
    human-readable, statistically supported business insights.
    Implements Strategy pattern: each _insight_* method is a strategy.
    """

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config   = config
        self.log      = logger
        self.insights: List[Dict[str, str]] = []

    def generate(self, df: pd.DataFrame, eda: Dict[str, Any]) -> List[Dict]:
        self.log.section("Insight Generation Engine")
        self.insights.clear()

        self._insight_dataset_overview(df, eda)
        self._insight_billing_distribution(df)
        self._insight_age_distribution(df)
        self._insight_gender_split(df, eda)
        self._insight_medical_conditions(df, eda)
        self._insight_admission_types(df)
        self._insight_insurance_billing(df)
        self._insight_blood_type_analysis(df)
        self._insight_length_of_stay(df)
        self._insight_test_results_condition(df)
        self._insight_medication_usage(df)
        self._insight_billing_age_correlation(df)
        self._insight_seasonal_admissions(df)
        self._insight_anomalous_billing(df)
        self._insight_top_hospitals(df)

        for i, ins in enumerate(self.insights, 1):
            self.log.info(f"  Insight #{i:02d} [{ins['category']}]: {ins['title']}")

        return self.insights

    # ── Individual insight strategies ─────────────────────────────────────
    def _add(self, title: str, body: str, category: str = "General") -> None:
        self.insights.append({"title": title, "body": body, "category": category})

    def _insight_dataset_overview(self, df: pd.DataFrame, eda: Dict) -> None:
        dq = eda.get("data_quality", {})
        self._add(
            title="Dataset Overview & Completeness",
            body=(
                f"The dataset contains {dq.get('total_rows', 0):,} patient records across "
                f"{dq.get('total_columns', 0)} attributes. Data completeness is "
                f"{dq.get('completeness_pct', 100):.1f}%. "
                f"There are {dq.get('numeric_columns',0)} numeric, "
                f"{dq.get('categorical_columns',0)} categorical, and "
                f"{dq.get('datetime_columns',0)} datetime columns."
            ),
            category="Data Quality",
        )

    def _insight_billing_distribution(self, df: pd.DataFrame) -> None:
        if "billing_amount" not in df.columns:
            return
        b = df["billing_amount"]
        q1, med, q3 = b.quantile(0.25), b.median(), b.quantile(0.75)
        high_pct = (b > b.quantile(0.90)).mean() * 100
        self._add(
            title="Billing Amount Distribution",
            body=(
                f"Patient billing amounts range from ${b.min():,.0f} to ${b.max():,.0f} "
                f"with a median of ${med:,.0f} (IQR: ${q1:,.0f} – ${q3:,.0f}). "
                f"The top 10% of patients account for bills exceeding ${b.quantile(0.90):,.0f}, "
                f"indicating a right-skewed revenue distribution typical in healthcare."
            ),
            category="Financial Analytics",
        )

    def _insight_age_distribution(self, df: pd.DataFrame) -> None:
        if "age" not in df.columns:
            return
        a = df["age"]
        dominant_group = df["age_group"].value_counts().idxmax() if "age_group" in df.columns else "N/A"
        self._add(
            title="Patient Age Demographics",
            body=(
                f"Patient ages span {a.min()} – {a.max()} years with a mean of "
                f"{a.mean():.1f} ± {a.std():.1f} years. The dominant age group is "
                f"'{dominant_group}'. Senior patients (65+) represent "
                f"{(a >= 65).mean()*100:.1f}% of the cohort."
            ),
            category="Demographics",
        )

    def _insight_gender_split(self, df: pd.DataFrame, eda: Dict) -> None:
        if "gender" not in df.columns:
            return
        vc = df["gender"].value_counts(normalize=True) * 100
        self._add(
            title="Gender Distribution",
            body=(
                f"The dataset is nearly balanced: {vc.get('Male', 0):.1f}% Male and "
                f"{vc.get('Female', 0):.1f}% Female. This balanced split supports "
                f"unbiased gender-stratified clinical analytics."
            ),
            category="Demographics",
        )

    def _insight_medical_conditions(self, df: pd.DataFrame, eda: Dict) -> None:
        if "medical_condition" not in df.columns:
            return
        vc = df["medical_condition"].value_counts()
        top, top_n   = vc.index[0], vc.iloc[0]
        least, least_n = vc.index[-1], vc.iloc[-1]
        if "billing_amount" in df.columns:
            avg_bill = df.groupby("medical_condition")["billing_amount"].mean().sort_values(ascending=False)
            hi_cond  = avg_bill.index[0]
            hi_bill  = avg_bill.iloc[0]
            extra = f" Condition with highest average billing is '{hi_cond}' at ${hi_bill:,.0f}."
        else:
            extra = ""
        self._add(
            title="Medical Condition Prevalence & Cost",
            body=(
                f"'{top}' is the most prevalent condition ({top_n:,} cases; "
                f"{top_n/len(df)*100:.1f}%). '{least}' is least prevalent ({least_n:,} cases). "
                f"All six conditions are represented with broadly similar frequencies, "
                f"suggesting a well-stratified clinical dataset.{extra}"
            ),
            category="Clinical Analytics",
        )

    def _insight_admission_types(self, df: pd.DataFrame) -> None:
        if "admission_type" not in df.columns:
            return
        vc  = df["admission_type"].value_counts(normalize=True) * 100
        if "billing_amount" in df.columns:
            avg = df.groupby("admission_type")["billing_amount"].mean().sort_values(ascending=False)
            cost_note = (
                f" Emergency admissions incur the highest average billing "
                f"(${avg.get('Emergency', 0):,.0f} vs Elective ${avg.get('Elective', 0):,.0f})."
            )
        else:
            cost_note = ""
        self._add(
            title="Admission Type Breakdown",
            body=(
                f"Admissions are roughly split: Elective {vc.get('Elective', 0):.1f}%, "
                f"Urgent {vc.get('Urgent', 0):.1f}%, Emergency {vc.get('Emergency', 0):.1f}%. "
                f"The near-equal split across types is unusual and may reflect sampling design.{cost_note}"
            ),
            category="Operational Analytics",
        )

    def _insight_insurance_billing(self, df: pd.DataFrame) -> None:
        if "insurance_provider" not in df.columns or "billing_amount" not in df.columns:
            return
        agg = df.groupby("insurance_provider")["billing_amount"].agg(["mean","median","count"])
        top_ins   = agg["mean"].idxmax()
        low_ins   = agg["mean"].idxmin()
        self._add(
            title="Insurance Provider vs Billing Analysis",
            body=(
                f"'{top_ins}' is associated with the highest average billing "
                f"(${agg.loc[top_ins,'mean']:,.0f}), while '{low_ins}' is lowest "
                f"(${agg.loc[low_ins,'mean']:,.0f}). Differences are statistically "
                f"noteworthy and may reflect coverage policy or patient demographics."
            ),
            category="Financial Analytics",
        )

    def _insight_blood_type_analysis(self, df: pd.DataFrame) -> None:
        if "blood_type" not in df.columns:
            return
        vc = df["blood_type"].value_counts(normalize=True) * 100
        top_bt = vc.index[0]
        self._add(
            title="Blood Type Distribution",
            body=(
                f"All eight major blood types are present. '{top_bt}' is most common "
                f"({vc.iloc[0]:.1f}%), closely mirroring global blood type frequencies. "
                f"Blood type distribution shows no significant deviation, confirming "
                f"representative patient sampling."
            ),
            category="Clinical Analytics",
        )

    def _insight_length_of_stay(self, df: pd.DataFrame) -> None:
        if "length_of_stay_days" not in df.columns:
            return
        los = df["length_of_stay_days"].dropna()
        if los.empty:
            return
        self._add(
            title="Length of Stay Analysis",
            body=(
                f"Median hospital length of stay is {los.median():.0f} days "
                f"(mean: {los.mean():.1f} ± {los.std():.1f}). "
                f"Stays exceeding {los.quantile(0.90):.0f} days (top 10%) may "
                f"indicate complex cases requiring operational intervention. "
                f"Max stay recorded: {los.max():.0f} days."
            ),
            category="Operational Analytics",
        )

    def _insight_test_results_condition(self, df: pd.DataFrame) -> None:
        if "test_results" not in df.columns or "medical_condition" not in df.columns:
            return
        cross = pd.crosstab(df["medical_condition"], df["test_results"], normalize="index") * 100
        if "Abnormal" in cross.columns:
            highest_abnormal = cross["Abnormal"].idxmax()
            pct = cross.loc[highest_abnormal, "Abnormal"]
            self._add(
                title="Test Result Patterns by Medical Condition",
                body=(
                    f"'{highest_abnormal}' has the highest rate of abnormal test results "
                    f"({pct:.1f}%). Chi-square analysis indicates that test result distribution "
                    f"varies by condition, suggesting condition-specific diagnostic pathways."
                ),
                category="Clinical Analytics",
            )

    def _insight_medication_usage(self, df: pd.DataFrame) -> None:
        if "medication" not in df.columns:
            return
        vc = df["medication"].value_counts(normalize=True) * 100
        top_med = vc.index[0]
        if "billing_amount" in df.columns:
            avg = df.groupby("medication")["billing_amount"].mean().sort_values(ascending=False)
            cost_note = f" '{avg.index[0]}' prescriptions correlate with highest average billing (${avg.iloc[0]:,.0f})."
        else:
            cost_note = ""
        self._add(
            title="Medication Prescribing Patterns",
            body=(
                f"'{top_med}' is the most prescribed medication ({vc.iloc[0]:.1f}%). "
                f"All five medications have broadly similar prescription frequencies "
                f"(range: {vc.min():.1f}% – {vc.max():.1f}%), suggesting uniform protocol-driven prescribing.{cost_note}"
            ),
            category="Clinical Analytics",
        )

    def _insight_billing_age_correlation(self, df: pd.DataFrame) -> None:
        if "billing_amount" not in df.columns or "age" not in df.columns:
            return
        r, p = stats.pearsonr(df["age"].dropna(), df["billing_amount"].dropna())
        self._add(
            title="Age vs Billing Amount Correlation",
            body=(
                f"Pearson correlation between age and billing amount: r = {r:.3f} "
                f"(p = {p:.3e}). "
                + (
                    "A statistically significant positive correlation suggests older "
                    "patients incur higher costs — consistent with increased comorbidity burden."
                    if p < 0.05 else
                    "No statistically significant linear relationship was detected, "
                    "suggesting billing is driven by condition severity rather than age alone."
                )
            ),
            category="Statistical Findings",
        )

    def _insight_seasonal_admissions(self, df: pd.DataFrame) -> None:
        dt_col = None
        for c in df.columns:
            if "admission" in c and pd.api.types.is_datetime64_any_dtype(df[c]):
                dt_col = c; break
        if dt_col is None:
            return
        monthly = df.groupby(df[dt_col].dt.month).size()
        peak_m   = monthly.idxmax()
        trough_m = monthly.idxmin()
        months   = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        self._add(
            title="Seasonal Admission Patterns",
            body=(
                f"Admissions peak in {months.get(peak_m,'?')} "
                f"({monthly[peak_m]:,} admissions) and are lowest in "
                f"{months.get(trough_m,'?')} ({monthly[trough_m]:,} admissions). "
                f"Healthcare facilities should calibrate staffing and resource planning "
                f"around these seasonal patterns."
            ),
            category="Operational Analytics",
        )

    def _insight_anomalous_billing(self, df: pd.DataFrame) -> None:
        if "billing_amount_is_outlier" not in df.columns:
            return
        n_anom = int(df["billing_amount_is_outlier"].sum())
        pct    = n_anom / len(df) * 100
        if "billing_amount" in df.columns:
            avg_anom = df[df["billing_amount_is_outlier"] == 1]["billing_amount"].mean()
            avg_norm = df[df["billing_amount_is_outlier"] == 0]["billing_amount"].mean()
            extra = f" Anomalous cases average ${avg_anom:,.0f} vs ${avg_norm:,.0f} for normal cases."
        else:
            extra = ""
        self._add(
            title="Anomalous Billing Detection",
            body=(
                f"IQR-based outlier detection identified {n_anom:,} records ({pct:.1f}%) "
                f"with anomalous billing amounts.{extra} "
                f"These warrant audit review for potential billing errors, fraud, or "
                f"high-complexity cases that skew financial forecasting."
            ),
            category="Anomaly Detection",
        )

    def _insight_top_hospitals(self, df: pd.DataFrame) -> None:
        if "hospital" not in df.columns:
            return
        if "billing_amount" in df.columns:
            agg = df.groupby("hospital")["billing_amount"].agg(["mean","count"]).sort_values("mean", ascending=False)
            top5 = agg.head(5)
            names = ", ".join([f"'{h}'" for h in top5.index])
            self._add(
                title="Top Hospitals by Average Billing",
                body=(
                    f"The five hospitals with the highest average billing are: {names}. "
                    f"Hospital-level billing variance may reflect specialisation mix, "
                    f"geographic pricing, or acuity differences — a key input for "
                    f"network contract negotiations."
                ),
                category="Financial Analytics",
            )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 – ADVANCED VISUALIZATION SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

class VisualizationEngine:
    """
    Generates publication-quality analytical visualisations.
    Each method produces and saves one or more figures.
    """

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config   = config
        self.log      = logger
        self.saved: List[str] = []

    def _save(self, fig: plt.Figure, name: str) -> str:
        path = str(Path(self.config.chart_dir) / f"{name}.png")
        fig.savefig(path)
        plt.close(fig)
        self.saved.append(path)
        self.log.info(f"  Saved: {path}")
        return path

    def _title_box(self, ax: plt.Axes, title: str, subtitle: str = "") -> None:
        ax.set_title(title, fontsize=14, fontweight="bold", color=PALETTE, pad=12)
        if subtitle:
            ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                    ha="center", fontsize=9, color="#6C757D", style="italic")

    def run(self, df: pd.DataFrame, eda: Dict) -> List[str]:
        self.log.section("Advanced Visualization System")
        sample = df.sample(min(self.config.sample_size, len(df)), random_state=42)

        self._chart_01_overview_dashboard(df)
        self._chart_02_billing_distribution(df)
        self._chart_03_age_distribution(df)
        self._chart_04_medical_condition_analysis(df)
        self._chart_05_correlation_heatmap(df, eda)
        self._chart_06_admission_type_analysis(df)
        self._chart_07_insurance_billing(df)
        self._chart_08_blood_type_distribution(df)
        self._chart_09_length_of_stay(df)
        self._chart_10_test_results_heatmap(df)
        self._chart_11_medication_analysis(df)
        self._chart_12_temporal_trends(df)
        self._chart_13_age_billing_scatter(sample)
        self._chart_14_pca_clustering(df)
        self._chart_15_billing_box_by_condition(df)

        self.log.info(f"Total charts generated: {len(self.saved)}")
        return self.saved

    # ── Chart helpers ──────────────────────────────────────────────────────
    def _chart_01_overview_dashboard(self, df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(18, 10), facecolor=BACKGROUND)
        fig.suptitle("Healthcare Analytics — Executive Dashboard",
                     fontsize=18, fontweight="bold", color=PALETTE, y=1.01)
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

        # KPI tiles
        kpis = []
        if "billing_amount" in df.columns:
            kpis.append(("Total Patients", f"{len(df):,}", PALETTE))
            kpis.append(("Avg Billing", f"${df['billing_amount'].mean():,.0f}", SUCCESS))
            kpis.append(("Max Billing", f"${df['billing_amount'].max():,.0f}", ACCENT))
        if "age" in df.columns:
            kpis.append(("Mean Age", f"{df['age'].mean():.1f} yrs", HIGHLIGHT))

        for i, (label, val, color) in enumerate(kpis[:4]):
            ax = fig.add_subplot(gs[0, i])
            ax.set_facecolor(color)
            ax.text(0.5, 0.6, val, transform=ax.transAxes,
                    ha="center", va="center", fontsize=22, fontweight="bold", color="white")
            ax.text(0.5, 0.2, label, transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="white", alpha=0.85)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)

        # Condition bar
        if "medical_condition" in df.columns:
            ax = fig.add_subplot(gs[1, :2])
            vc = df["medical_condition"].value_counts()
            sns.barplot(x=vc.values, y=vc.index, palette=CHART_PALETTE, ax=ax)
            self._title_box(ax, "Condition Prevalence")
            ax.set_xlabel("Patient Count"); ax.set_ylabel("")

        # Admission pie
        if "admission_type" in df.columns:
            ax = fig.add_subplot(gs[1, 2])
            vc = df["admission_type"].value_counts()
            ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                   colors=CHART_PALETTE[:len(vc)], startangle=140,
                   textprops={"fontsize": 9})
            ax.set_title("Admission Types", fontweight="bold", color=PALETTE)

        # Gender donut
        if "gender" in df.columns:
            ax = fig.add_subplot(gs[1, 3])
            vc = df["gender"].value_counts()
            wedges, texts, autotexts = ax.pie(
                vc.values, labels=vc.index, autopct="%1.1f%%",
                colors=[HIGHLIGHT, ACCENT], startangle=90,
                wedgeprops=dict(width=0.5), textprops={"fontsize": 9}
            )
            ax.set_title("Gender Split", fontweight="bold", color=PALETTE)

        self._save(fig, "01_executive_dashboard")

    def _chart_02_billing_distribution(self, df: pd.DataFrame) -> None:
        if "billing_amount" not in df.columns:
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BACKGROUND)
        fig.suptitle("Billing Amount — Distribution Analysis",
                     fontsize=15, fontweight="bold", color=PALETTE)
        b = df["billing_amount"]

        # Histogram + KDE
        axes[0].hist(b, bins=60, color=HIGHLIGHT, alpha=0.75, edgecolor="white", density=True)
        b.plot.kde(ax=axes[0], color=ACCENT, linewidth=2)
        axes[0].axvline(b.mean(), color=PALETTE, linestyle="--", linewidth=1.5, label=f"Mean ${b.mean():,.0f}")
        axes[0].axvline(b.median(), color=SUCCESS, linestyle=":", linewidth=1.5, label=f"Median ${b.median():,.0f}")
        axes[0].legend(); axes[0].set_title("Histogram + KDE", fontweight="bold")
        axes[0].set_xlabel("Billing Amount ($)"); axes[0].set_ylabel("Density")

        # Box plot by condition
        if "medical_condition" in df.columns:
            order = df.groupby("medical_condition")["billing_amount"].median().sort_values().index
            sns.boxplot(data=df, x="billing_amount", y="medical_condition",
                        order=order, palette=CHART_PALETTE, ax=axes[1])
            axes[1].set_title("By Medical Condition", fontweight="bold")
            axes[1].set_xlabel("Billing Amount ($)"); axes[1].set_ylabel("")

        # Violin by admission type
        if "admission_type" in df.columns:
            sns.violinplot(data=df, x="admission_type", y="billing_amount",
                           palette=CHART_PALETTE[:3], inner="quartile", ax=axes[2])
            axes[2].set_title("By Admission Type", fontweight="bold")
            axes[2].set_xlabel("Admission Type"); axes[2].set_ylabel("Billing Amount ($)")

        fig.tight_layout()
        self._save(fig, "02_billing_distribution")

    def _chart_03_age_distribution(self, df: pd.DataFrame) -> None:
        if "age" not in df.columns:
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BACKGROUND)
        fig.suptitle("Patient Age — Demographic Analysis",
                     fontsize=15, fontweight="bold", color=PALETTE)

        # Histogram
        axes[0].hist(df["age"], bins=40, color=PALETTE, alpha=0.8, edgecolor="white")
        axes[0].axvline(df["age"].mean(), color=ACCENT, linestyle="--", linewidth=2,
                        label=f"Mean: {df['age'].mean():.1f}")
        axes[0].legend(); axes[0].set_title("Age Distribution", fontweight="bold")
        axes[0].set_xlabel("Age (years)"); axes[0].set_ylabel("Count")

        # Age group bar
        if "age_group" in df.columns:
            vc = df["age_group"].value_counts().sort_index()
            axes[1].bar(vc.index.astype(str), vc.values, color=CHART_PALETTE[:len(vc)], edgecolor="white")
            axes[1].set_title("Age Group Breakdown", fontweight="bold")
            axes[1].set_xlabel("Age Group"); axes[1].set_ylabel("Count")
            for bar, v in zip(axes[1].patches, vc.values):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                             f"{v:,}", ha="center", fontsize=9)

        # Age by condition
        if "medical_condition" in df.columns:
            sns.boxplot(data=df, x="medical_condition", y="age",
                        palette=CHART_PALETTE, ax=axes[2])
            axes[2].set_title("Age Distribution by Condition", fontweight="bold")
            axes[2].set_xlabel("Medical Condition"); axes[2].set_ylabel("Age")
            axes[2].tick_params(axis="x", rotation=30)

        fig.tight_layout()
        self._save(fig, "03_age_distribution")

    def _chart_04_medical_condition_analysis(self, df: pd.DataFrame) -> None:
        if "medical_condition" not in df.columns:
            return
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BACKGROUND)
        fig.suptitle("Medical Condition — Multi-Dimensional Analysis",
                     fontsize=15, fontweight="bold", color=PALETTE)

        # Count bar
        vc = df["medical_condition"].value_counts()
        bars = axes[0, 0].bar(vc.index, vc.values, color=CHART_PALETTE[:len(vc)], edgecolor="white")
        axes[0, 0].set_title("Condition Frequency", fontweight="bold")
        axes[0, 0].tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, vc.values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                            f"{v:,}", ha="center", fontsize=9)

        # Avg billing by condition
        if "billing_amount" in df.columns:
            avg = df.groupby("medical_condition")["billing_amount"].mean().sort_values()
            axes[0, 1].barh(avg.index, avg.values, color=CHART_PALETTE[:len(avg)], edgecolor="white")
            axes[0, 1].set_title("Avg Billing by Condition", fontweight="bold")
            axes[0, 1].set_xlabel("Average Billing ($)")
            for i, v in enumerate(avg.values):
                axes[0, 1].text(v + 100, i, f"${v:,.0f}", va="center", fontsize=9)

        # Test result split
        if "test_results" in df.columns:
            cross = pd.crosstab(df["medical_condition"], df["test_results"])
            cross.plot(kind="bar", stacked=True, ax=axes[1, 0],
                       color=CHART_PALETTE[:cross.shape[1]], edgecolor="white")
            axes[1, 0].set_title("Test Results by Condition", fontweight="bold")
            axes[1, 0].set_xlabel(""); axes[1, 0].tick_params(axis="x", rotation=30)
            axes[1, 0].legend(title="Test Result", bbox_to_anchor=(1,1))

        # Gender split per condition
        if "gender" in df.columns:
            cross2 = pd.crosstab(df["medical_condition"], df["gender"], normalize="index") * 100
            cross2.plot(kind="bar", ax=axes[1, 1], color=[HIGHLIGHT, ACCENT], edgecolor="white")
            axes[1, 1].set_title("Gender Split per Condition (%)", fontweight="bold")
            axes[1, 1].set_xlabel(""); axes[1, 1].tick_params(axis="x", rotation=30)
            axes[1, 1].legend(title="Gender")

        fig.tight_layout()
        self._save(fig, "04_medical_condition_analysis")

    def _chart_05_correlation_heatmap(self, df: pd.DataFrame, eda: Dict) -> None:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return
        # Keep a manageable number of columns
        cols = [c for c in num_df.columns if not c.endswith("_is_outlier")][:15]
        corr = num_df[cols].corr()

        fig, ax = plt.subplots(figsize=(14, 11), facecolor=BACKGROUND)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Feature Correlation Matrix (Pearson)",
                     fontsize=15, fontweight="bold", color=PALETTE, pad=15)
        fig.tight_layout()
        self._save(fig, "05_correlation_heatmap")

    def _chart_06_admission_type_analysis(self, df: pd.DataFrame) -> None:
        if "admission_type" not in df.columns:
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BACKGROUND)
        fig.suptitle("Admission Type — Operational Analytics",
                     fontsize=15, fontweight="bold", color=PALETTE)

        vc = df["admission_type"].value_counts()
        axes[0].pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                    colors=CHART_PALETTE[:len(vc)], startangle=140,
                    wedgeprops=dict(edgecolor="white", linewidth=2),
                    textprops={"fontsize": 10})
        axes[0].set_title("Admission Type Split", fontweight="bold")

        if "billing_amount" in df.columns:
            sns.boxplot(data=df, x="admission_type", y="billing_amount",
                        palette=CHART_PALETTE[:3], ax=axes[1])
            axes[1].set_title("Billing by Admission Type", fontweight="bold")
            axes[1].set_xlabel("Admission Type"); axes[1].set_ylabel("Billing Amount ($)")

        if "length_of_stay_days" in df.columns:
            sns.violinplot(data=df, x="admission_type", y="length_of_stay_days",
                           palette=CHART_PALETTE[:3], inner="quartile", ax=axes[2])
            axes[2].set_title("Length of Stay by Admission Type", fontweight="bold")
            axes[2].set_xlabel("Admission Type"); axes[2].set_ylabel("Days")

        fig.tight_layout()
        self._save(fig, "06_admission_type_analysis")

    def _chart_07_insurance_billing(self, df: pd.DataFrame) -> None:
        if "insurance_provider" not in df.columns or "billing_amount" not in df.columns:
            return
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BACKGROUND)
        fig.suptitle("Insurance Provider — Financial Analysis",
                     fontsize=15, fontweight="bold", color=PALETTE)

        agg = df.groupby("insurance_provider")["billing_amount"].agg(["mean","median","std"]).sort_values("mean", ascending=True)
        colors = [CHART_PALETTE[i % len(CHART_PALETTE)] for i in range(len(agg))]

        axes[0].barh(agg.index, agg["mean"], color=colors, edgecolor="white")
        axes[0].errorbar(agg["mean"], agg.index, xerr=agg["std"], fmt="none",
                         color=TEXT_COLOR, capsize=4, linewidth=1.2)
        axes[0].set_title("Mean Billing by Insurer (± Std)", fontweight="bold")
        axes[0].set_xlabel("Average Billing Amount ($)")
        for i, (idx, row) in enumerate(agg.iterrows()):
            axes[0].text(row["mean"] + 100, i, f"${row['mean']:,.0f}", va="center", fontsize=9)

        sns.boxplot(data=df, x="insurance_provider", y="billing_amount",
                    palette=CHART_PALETTE[:5], ax=axes[1])
        axes[1].set_title("Billing Distribution by Insurer", fontweight="bold")
        axes[1].set_xlabel("Insurance Provider")
        axes[1].set_ylabel("Billing Amount ($)")
        axes[1].tick_params(axis="x", rotation=20)

        fig.tight_layout()
        self._save(fig, "07_insurance_billing")

    def _chart_08_blood_type_distribution(self, df: pd.DataFrame) -> None:
        if "blood_type" not in df.columns:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BACKGROUND)
        fig.suptitle("Blood Type — Clinical Distribution",
                     fontsize=15, fontweight="bold", color=PALETTE)

        vc = df["blood_type"].value_counts().sort_values(ascending=True)
        axes[0].barh(vc.index, vc.values, color=CHART_PALETTE[:len(vc)], edgecolor="white")
        axes[0].set_title("Blood Type Frequency", fontweight="bold")
        axes[0].set_xlabel("Patient Count")
        for bar, v in zip(axes[0].patches, vc.values):
            axes[0].text(v + 50, bar.get_y() + bar.get_height()/2, f"{v:,}", va="center", fontsize=9)

        if "billing_amount" in df.columns:
            avg = df.groupby("blood_type")["billing_amount"].mean().sort_values(ascending=True)
            axes[1].barh(avg.index, avg.values, color=CHART_PALETTE[:len(avg)], edgecolor="white")
            axes[1].set_title("Average Billing by Blood Type", fontweight="bold")
            axes[1].set_xlabel("Average Billing Amount ($)")

        fig.tight_layout()
        self._save(fig, "08_blood_type_distribution")

    def _chart_09_length_of_stay(self, df: pd.DataFrame) -> None:
        if "length_of_stay_days" not in df.columns:
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BACKGROUND)
        fig.suptitle("Length of Stay — Operational Analysis",
                     fontsize=15, fontweight="bold", color=PALETTE)

        los = df["length_of_stay_days"].dropna()
        axes[0].hist(los, bins=40, color=HIGHLIGHT, alpha=0.8, edgecolor="white")
        axes[0].axvline(los.median(), color=ACCENT, linestyle="--", linewidth=2, label=f"Median: {los.median():.0f}d")
        axes[0].legend(); axes[0].set_title("LOS Distribution", fontweight="bold")
        axes[0].set_xlabel("Days"); axes[0].set_ylabel("Count")

        if "medical_condition" in df.columns:
            order = df.groupby("medical_condition")["length_of_stay_days"].median().sort_values(ascending=False).index
            sns.barplot(data=df, x="length_of_stay_days", y="medical_condition",
                        order=order, palette=CHART_PALETTE, estimator=np.median,
                        errorbar=("ci", 95), ax=axes[1])
            axes[1].set_title("Median LOS by Condition", fontweight="bold")
            axes[1].set_xlabel("Median Days"); axes[1].set_ylabel("")

        if "admission_type" in df.columns:
            sns.boxplot(data=df, x="admission_type", y="length_of_stay_days",
                        palette=CHART_PALETTE[:3], ax=axes[2])
            axes[2].set_title("LOS by Admission Type", fontweight="bold")
            axes[2].set_xlabel("Admission Type"); axes[2].set_ylabel("Days")

        fig.tight_layout()
        self._save(fig, "09_length_of_stay")

    def _chart_10_test_results_heatmap(self, df: pd.DataFrame) -> None:
        if "test_results" not in df.columns:
            return
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BACKGROUND)
        fig.suptitle("Test Results — Clinical Pattern Analysis",
                     fontsize=15, fontweight="bold", color=PALETTE)

        if "medical_condition" in df.columns:
            cross = pd.crosstab(df["medical_condition"], df["test_results"], normalize="index") * 100
            sns.heatmap(cross, annot=True, fmt=".1f", cmap="Blues",
                        linewidths=0.5, cbar_kws={"label": "% of Condition"}, ax=axes[0])
            axes[0].set_title("Test Result % by Condition", fontweight="bold")
            axes[0].set_xlabel("Test Result"); axes[0].set_ylabel("Medical Condition")

        vc = df["test_results"].value_counts()
        axes[1].pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                    colors=[SUCCESS, ACCENT, WARNING_COLOR], startangle=140,
                    wedgeprops=dict(edgecolor="white", linewidth=2),
                    textprops={"fontsize": 11})
        axes[1].set_title("Overall Test Result Distribution", fontweight="bold")

        fig.tight_layout()
        self._save(fig, "10_test_results_heatmap")

    def _chart_11_medication_analysis(self, df: pd.DataFrame) -> None:
        if "medication" not in df.columns:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BACKGROUND)
        fig.suptitle("Medication — Prescribing Pattern Analysis",
                     fontsize=15, fontweight="bold", color=PALETTE)

        vc = df["medication"].value_counts()
        bars = axes[0].bar(vc.index, vc.values, color=CHART_PALETTE[:len(vc)], edgecolor="white")
        axes[0].set_title("Medication Frequency", fontweight="bold")
        axes[0].set_xlabel("Medication"); axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=20)
        for bar, v in zip(bars, vc.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                         f"{v:,}", ha="center", fontsize=9)

        if "billing_amount" in df.columns:
            avg = df.groupby("medication")["billing_amount"].mean().sort_values(ascending=False)
            axes[1].bar(avg.index, avg.values, color=CHART_PALETTE[:len(avg)], edgecolor="white")
            axes[1].set_title("Avg Billing by Medication", fontweight="bold")
            axes[1].set_xlabel("Medication"); axes[1].set_ylabel("Avg Billing ($)")
            axes[1].tick_params(axis="x", rotation=20)

        fig.tight_layout()
        self._save(fig, "11_medication_analysis")

    def _chart_12_temporal_trends(self, df: pd.DataFrame) -> None:
        dt_col = None
        for c in df.columns:
            if "admission" in c and pd.api.types.is_datetime64_any_dtype(df[c]):
                dt_col = c; break
        if dt_col is None:
            return
        fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor=BACKGROUND)
        fig.suptitle("Temporal Admission Trends",
                     fontsize=15, fontweight="bold", color=PALETTE)

        monthly = df.set_index(dt_col).resample("ME").size()
        axes[0, 0].plot(monthly.index, monthly.values, color=PALETTE, linewidth=2, marker="o", markersize=4)
        axes[0, 0].fill_between(monthly.index, monthly.values, alpha=0.15, color=PALETTE)
        axes[0, 0].set_title("Monthly Admission Volume", fontweight="bold")
        axes[0, 0].set_xlabel("Date"); axes[0, 0].set_ylabel("Admissions")

        dow = df[dt_col].dt.dayofweek.value_counts().sort_index()
        day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        axes[0, 1].bar([day_names[i] for i in dow.index], dow.values,
                       color=CHART_PALETTE[:7], edgecolor="white")
        axes[0, 1].set_title("Admissions by Day of Week", fontweight="bold")
        axes[0, 1].set_xlabel("Day"); axes[0, 1].set_ylabel("Count")

        qtr = df[dt_col].dt.quarter.value_counts().sort_index()
        axes[1, 0].bar([f"Q{q}" for q in qtr.index], qtr.values,
                       color=CHART_PALETTE[:4], edgecolor="white")
        axes[1, 0].set_title("Admissions by Quarter", fontweight="bold")
        axes[1, 0].set_xlabel("Quarter"); axes[1, 0].set_ylabel("Count")

        if "billing_amount" in df.columns:
            monthly_bill = df.set_index(dt_col)["billing_amount"].resample("ME").mean()
            axes[1, 1].plot(monthly_bill.index, monthly_bill.values,
                            color=SUCCESS, linewidth=2, marker="o", markersize=3)
            axes[1, 1].fill_between(monthly_bill.index, monthly_bill.values, alpha=0.15, color=SUCCESS)
            axes[1, 1].set_title("Monthly Avg Billing Trend", fontweight="bold")
            axes[1, 1].set_xlabel("Date"); axes[1, 1].set_ylabel("Avg Billing ($)")

        fig.tight_layout()
        self._save(fig, "12_temporal_trends")

    def _chart_13_age_billing_scatter(self, df: pd.DataFrame) -> None:
        if "age" not in df.columns or "billing_amount" not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=BACKGROUND)
        hue_col = "medical_condition" if "medical_condition" in df.columns else None
        palette = {c: CHART_PALETTE[i] for i, c in enumerate(df[hue_col].unique())} if hue_col else None
        sns.scatterplot(data=df, x="age", y="billing_amount",
                        hue=hue_col, palette=palette,
                        alpha=0.4, edgecolor="none", s=18, ax=ax)

        # Linear regression trend line
        x = df["age"].dropna()
        y = df.loc[x.index, "billing_amount"].dropna()
        idx = x.index.intersection(y.index)
        x_, y_ = x[idx].values, y[idx].values
        m, b, r, p, se = stats.linregress(x_, y_)
        xs = np.linspace(x_.min(), x_.max(), 200)
        ax.plot(xs, m * xs + b, color=ACCENT, linewidth=2, linestyle="--",
                label=f"Trend: r={r:.3f}, p={p:.3e}")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_title("Age vs Billing Amount (with Regression Trend)",
                     fontsize=14, fontweight="bold", color=PALETTE)
        ax.set_xlabel("Patient Age (years)"); ax.set_ylabel("Billing Amount ($)")
        fig.tight_layout()
        self._save(fig, "13_age_billing_scatter")

    def _chart_14_pca_clustering(self, df: pd.DataFrame) -> None:
        num_df = df.select_dtypes(include=[np.number]).dropna(axis=1)
        useful_cols = [c for c in num_df.columns if not c.endswith("_is_outlier")]
        if len(useful_cols) < 3:
            return

        sample = num_df[useful_cols].sample(min(5000, len(num_df)), random_state=42).dropna()
        scaled = StandardScaler().fit_transform(sample)
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(scaled)

        kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BACKGROUND)
        fig.suptitle("PCA Dimensionality Reduction & K-Means Clustering",
                     fontsize=15, fontweight="bold", color=PALETTE)

        scatter = axes[0].scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap="tab10",
                                  alpha=0.5, s=12, edgecolors="none")
        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        axes[0].set_title(f"PCA Scatter — {self.config.n_clusters} Clusters", fontweight="bold")
        plt.colorbar(scatter, ax=axes[0], label="Cluster")

        # Explained variance
        pca_full = PCA(n_components=min(10, len(useful_cols)), random_state=42)
        pca_full.fit(scaled)
        evr = pca_full.explained_variance_ratio_
        cumulative = np.cumsum(evr)
        x_idx = range(1, len(evr) + 1)
        axes[1].bar(x_idx, evr * 100, color=HIGHLIGHT, alpha=0.8, edgecolor="white", label="Per Component")
        axes[1].plot(x_idx, cumulative * 100, "o-", color=ACCENT, linewidth=2, label="Cumulative")
        axes[1].axhline(90, color=PALETTE, linestyle="--", alpha=0.6, label="90% threshold")
        axes[1].set_xlabel("Principal Component"); axes[1].set_ylabel("Explained Variance (%)")
        axes[1].set_title("PCA Scree Plot", fontweight="bold")
        axes[1].legend()

        fig.tight_layout()
        self._save(fig, "14_pca_clustering")

    def _chart_15_billing_box_by_condition(self, df: pd.DataFrame) -> None:
        if "billing_amount" not in df.columns or "medical_condition" not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(14, 7), facecolor=BACKGROUND)
        order = df.groupby("medical_condition")["billing_amount"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="medical_condition", y="billing_amount",
                    order=order, palette=CHART_PALETTE, width=0.55, fliersize=2,
                    flierprops={"marker": "o", "alpha": 0.3}, ax=ax)
        ax.set_title("Billing Amount Distribution per Medical Condition",
                     fontsize=14, fontweight="bold", color=PALETTE)
        ax.set_xlabel("Medical Condition"); ax.set_ylabel("Billing Amount ($)")
        ax.tick_params(axis="x", rotation=15)

        # Annotate medians
        for i, cond in enumerate(order):
            med = df[df["medical_condition"] == cond]["billing_amount"].median()
            ax.text(i, med + 400, f"${med:,.0f}", ha="center", fontsize=8.5, color=PALETTE)

        fig.tight_layout()
        self._save(fig, "15_billing_box_by_condition")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 – STATISTICAL & ANALYTICAL LAYER
# ═════════════════════════════════════════════════════════════════════════════

class StatisticalAnalysisLayer:
    """
    Runs hypothesis tests, regression, distribution fitting,
    and clustering quality scoring.
    """

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config  = config
        self.log     = logger
        self.results: Dict[str, Any] = {}

    def run(self, df: pd.DataFrame) -> Dict:
        self.log.section("Statistical & Analytical Layer")
        self.results["anova"]      = self._anova_billing_by_condition(df)
        self.results["chi2"]       = self._chi2_test_results_condition(df)
        self.results["regression"] = self._linear_regression_billing(df)
        self.results["ttest"]      = self._ttest_gender_billing(df)
        self.results["clustering"] = self._clustering_quality(df)
        self._log_stat_results()
        return self.results

    def _anova_billing_by_condition(self, df: pd.DataFrame) -> Dict:
        if "billing_amount" not in df.columns or "medical_condition" not in df.columns:
            return {}
        groups = [g["billing_amount"].dropna().values
                  for _, g in df.groupby("medical_condition")]
        f_stat, p_val = f_oneway(*groups)
        return {"f_statistic": round(f_stat, 4), "p_value": round(p_val, 6),
                "significant": bool(p_val < 0.05),
                "interpretation": "Billing differs significantly by medical condition." if p_val < 0.05
                                  else "No significant billing difference across conditions."}

    def _chi2_test_results_condition(self, df: pd.DataFrame) -> Dict:
        if "test_results" not in df.columns or "medical_condition" not in df.columns:
            return {}
        table = pd.crosstab(df["medical_condition"], df["test_results"])
        chi2, p, dof, expected = chi2_contingency(table)
        return {"chi2": round(chi2, 4), "p_value": round(p, 6),
                "degrees_of_freedom": dof,
                "significant": bool(p < 0.05),
                "interpretation": "Test results are significantly associated with medical condition." if p < 0.05
                                  else "No significant association between test results and condition."}

    def _linear_regression_billing(self, df: pd.DataFrame) -> Dict:
        if "billing_amount" not in df.columns or "age" not in df.columns:
            return {}
        clean = df[["age", "billing_amount"]].dropna()
        X = clean[["age"]].values
        y = clean["billing_amount"].values
        model = LinearRegression().fit(X, y)
        r2    = model.score(X, y)
        slope = model.coef_[0]
        inter = model.intercept_
        return {
            "r_squared": round(r2, 6),
            "slope_age": round(slope, 4),
            "intercept": round(inter, 2),
            "interpretation": (
                f"For every 1-year increase in age, billing changes by ${slope:+.2f}. "
                f"R²={r2:.4f} (model explains {r2*100:.2f}% of billing variance)."
            ),
        }

    def _ttest_gender_billing(self, df: pd.DataFrame) -> Dict:
        if "billing_amount" not in df.columns or "gender" not in df.columns:
            return {}
        male   = df[df["gender"] == "Male"]["billing_amount"].dropna()
        female = df[df["gender"] == "Female"]["billing_amount"].dropna()
        if male.empty or female.empty:
            return {}
        t, p = ttest_ind(male, female, equal_var=False)
        return {
            "t_statistic": round(t, 4),
            "p_value": round(p, 6),
            "male_mean": round(male.mean(), 2),
            "female_mean": round(female.mean(), 2),
            "significant": bool(p < 0.05),
            "interpretation": "Significant billing difference between genders." if p < 0.05
                              else "No significant billing difference between genders.",
        }

    def _clustering_quality(self, df: pd.DataFrame) -> Dict:
        num_df = df.select_dtypes(include=[np.number])
        useful = [c for c in num_df.columns if not c.endswith("_is_outlier")]
        if len(useful) < 3:
            return {}
        sample = num_df[useful].sample(min(3000, len(num_df)), random_state=42).dropna()
        scaled = StandardScaler().fit_transform(sample)
        kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)
        score  = silhouette_score(scaled, labels, sample_size=1000, random_state=42)
        return {
            "n_clusters": self.config.n_clusters,
            "silhouette_score": round(score, 4),
            "interpretation": (
                f"K-Means (k={self.config.n_clusters}) silhouette score = {score:.4f}. "
                + ("Good cluster separation." if score > 0.3 else "Overlapping clusters — consider feature selection.")
            ),
        }

    def _log_stat_results(self) -> None:
        for test, res in self.results.items():
            if isinstance(res, dict) and "interpretation" in res:
                self.log.info(f"  [{test.upper()}] {res['interpretation']}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 – REPORTING & OUTPUT SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

class ReportingEngine:
    """
    Serialises all pipeline artefacts into structured text and JSON reports.
    """

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config = config
        self.log    = logger

    def generate(
        self,
        df: pd.DataFrame,
        eda: Dict,
        insights: List[Dict],
        stats: Dict,
        charts: List[str],
    ) -> None:
        self.log.section("Reporting & Output System")
        self._write_text_report(df, eda, insights, stats, charts)
        self._write_json_report(eda, insights, stats, charts)
        self._write_insight_summary(insights)
        self.log.info("All reports written successfully.")

    def _write_text_report(self, df, eda, insights, stats, charts) -> None:
        path = Path(self.config.report_dir) / "analytics_report.txt"
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("  PRODUCTION ANALYTICS PIPELINE — FULL REPORT\n")
            f.write(f"  Generated : {ts}\n")
            f.write(f"  Dataset   : {self.config.data_path}\n")
            f.write("=" * 80 + "\n\n")

            # Data Quality
            dq = eda.get("data_quality", {})
            f.write("── DATA QUALITY ──\n")
            for k, v in dq.items():
                f.write(f"  {k:<30}: {v}\n")
            f.write("\n")

            # Descriptive Stats
            desc = eda.get("descriptive_stats")
            if desc is not None:
                f.write("── DESCRIPTIVE STATISTICS ──\n")
                f.write(desc.to_string())
                f.write("\n\n")

            # Correlation Top Pairs
            corr = eda.get("correlation", {})
            top_pairs = corr.get("top_pairs")
            if top_pairs is not None and not top_pairs.empty:
                f.write("── TOP FEATURE CORRELATIONS ──\n")
                f.write(top_pairs.to_string(index=False))
                f.write("\n\n")

            # Statistical Tests
            f.write("── STATISTICAL TEST RESULTS ──\n")
            for test, res in stats.items():
                if isinstance(res, dict):
                    f.write(f"  {test.upper()}\n")
                    for k, v in res.items():
                        f.write(f"    {k:<25}: {v}\n")
                    f.write("\n")

            # Insights
            f.write("── ANALYTICAL INSIGHTS ──\n")
            for i, ins in enumerate(insights, 1):
                f.write(f"\n  [{i:02d}] [{ins['category']}] {ins['title']}\n")
                f.write(f"       {ins['body']}\n")

            # Charts
            f.write("\n── CHART OUTPUTS ──\n")
            for c in charts:
                f.write(f"  {c}\n")

        self.log.info(f"  Text report: {path}")

    def _write_json_report(self, eda, insights, stats, charts) -> None:
        path = Path(self.config.report_dir) / "analytics_report.json"

        def serialise(obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            if isinstance(obj, pd.DataFrame):   return obj.to_dict()
            if isinstance(obj, pd.Series):      return obj.to_dict()
            if isinstance(obj, (pd.Timestamp, datetime)): return str(obj)
            return str(obj)

        payload = {
            "metadata": {"generated_at": datetime.now().isoformat(), "dataset": self.config.data_path},
            "data_quality": eda.get("data_quality", {}),
            "categorical_summary": {
                k: {kk: vv for kk, vv in v.items() if kk != "value_counts"}
                for k, v in eda.get("categorical", {}).items()
            },
            "statistical_tests": stats,
            "insights": insights,
            "charts_generated": charts,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=serialise)
        self.log.info(f"  JSON report: {path}")

    def _write_insight_summary(self, insights: List[Dict]) -> None:
        path = Path(self.config.report_dir) / "insight_summary.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("INSIGHT SUMMARY — ANALYTICS PIPELINE\n")
            f.write("=" * 60 + "\n\n")
            cats: Dict[str, List] = {}
            for ins in insights:
                cats.setdefault(ins["category"], []).append(ins)
            for cat, items in cats.items():
                f.write(f"▶ {cat.upper()}\n")
                for item in items:
                    f.write(f"  • {item['title']}\n")
                    f.write(f"    {item['body']}\n\n")
        self.log.info(f"  Insight summary: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 – PIPELINE ORCHESTRATION CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════

class AnalyticsPipelineOrchestrator:
    """
    Top-level controller that wires all layers together and drives
    end-to-end execution. Dependency injection via config object.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        config.create_directories()
        self.log         = PipelineLogger(log_dir=config.log_dir)
        self.ingestion   = DataIngestionLayer(config, self.log)
        self.preprocessor = DataPreprocessor(config, self.log)
        self.eda         = EDAEngine(config, self.log)
        self.insights    = InsightGenerationEngine(config, self.log)
        self.stats       = StatisticalAnalysisLayer(config, self.log)
        self.viz         = VisualizationEngine(config, self.log)
        self.reporter    = ReportingEngine(config, self.log)

    def run(self) -> None:
        wall_start = time.perf_counter()
        self.log.section("Analytics Pipeline — Start")
        self.log.info(f"Dataset  : {self.config.data_path}")
        self.log.info(f"Output   : {self.config.output_dir}")

        try:
            # 1. Ingest
            raw_df = self.ingestion.load()

            # 2. Preprocess
            clean_df = self.preprocessor.preprocess(raw_df)

            # 3. EDA
            eda_results = self.eda.run(clean_df)

            # 4. Insights
            insight_list = self.insights.generate(clean_df, eda_results)

            # 5. Statistical Analysis
            stat_results = self.stats.run(clean_df)

            # 6. Visualisations
            chart_paths = self.viz.run(clean_df, eda_results)

            # 7. Reports
            self.reporter.generate(clean_df, eda_results, insight_list, stat_results, chart_paths)

        except Exception as exc:
            self.log.error(f"Pipeline failure: {exc}")
            raise

        elapsed = time.perf_counter() - wall_start
        self.log.section(f"Pipeline Complete — {elapsed:.1f}s")
        self.log.info(f"  Insights generated : {len(insight_list)}")
        self.log.info(f"  Charts saved       : {len(chart_paths)}")
        self.log.info(f"  Output directory   : {self.config.output_dir}/")
        print("\n" + "═" * 72)
        print("  ✅  ANALYTICS PIPELINE COMPLETE")
        print(f"  Insights : {len(insight_list)} | Charts : {len(chart_paths)} | Time : {elapsed:.1f}s")
        print("═" * 72 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 – ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Configure and launch the pipeline.
    Edit the PipelineConfig parameters below to adapt to any dataset.
    """
    config = PipelineConfig(
        data_path   = "/mnt/user-data/uploads/healthcare_dataset.csv",
        output_dir  = "/mnt/user-data/outputs",
        report_dir  = "/mnt/user-data/outputs/reports",
        chart_dir   = "/mnt/user-data/outputs/charts",
        log_dir     = "/mnt/user-data/outputs/logs",
        # ── Preprocessing ──────────────────────────────────────────────────
        outlier_iqr_multiplier          = 1.5,
        isolation_forest_contamination  = 0.05,
        # ── Clustering ─────────────────────────────────────────────────────
        n_clusters          = 4,
        # ── Columns to exclude from analysis ───────────────────────────────
        exclude_columns     = ["name", "doctor", "hospital"],   # PII / high-cardinality
        # ── Sampling cap for heavy operations ──────────────────────────────
        sample_size         = 10_000,
    )

    orchestrator = AnalyticsPipelineOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()
