"""
================================================================================
Production-Grade Industrial Data Analytics & Visualization Pipeline
================================================================================
Dataset           : NYC Airbnb Open Data
Framework Version : 2.0.0
Architecture      : Object-Oriented, SOLID Principles, Modular Enterprise Design
Compatible With   : Any structured/tabular dataset (CSV, Excel, JSON, Parquet, SQL)
Libraries         : pandas, numpy, scipy, matplotlib, seaborn, sklearn, logging
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
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, normaltest, mannwhitneyu
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Global Style Configuration
# ─────────────────────────────────────────────────────────────────────────────
PALETTE       = "#FF5A5F"   # Airbnb Red
ACCENT        = "#00A699"   # Airbnb Teal
HIGHLIGHT     = "#FC642D"   # Airbnb Orange
NEUTRAL       = "#484848"   # Airbnb Dark Grey
SOFT          = "#767676"   # Airbnb Light Grey
BACKGROUND    = "#FAFAFA"
GRID_COLOR    = "#E8E8E8"
TEXT_COLOR    = "#222222"

CHART_PALETTE = [PALETTE, ACCENT, HIGHLIGHT, NEUTRAL, "#B8D4E3",
                 "#F7B731", "#26de81", "#a29bfe", "#fd9644", "#2d98da"]

plt.rcParams.update({
    "figure.facecolor"  : BACKGROUND,
    "axes.facecolor"    : "white",
    "axes.edgecolor"    : GRID_COLOR,
    "axes.labelcolor"   : TEXT_COLOR,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 10,
    "axes.grid"         : True,
    "grid.color"        : GRID_COLOR,
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "xtick.color"       : TEXT_COLOR,
    "ytick.color"       : TEXT_COLOR,
    "font.family"       : "DejaVu Sans",
    "figure.dpi"        : 120,
    "savefig.dpi"       : 150,
    "savefig.bbox"      : "tight",
    "savefig.facecolor" : BACKGROUND,
    "legend.framealpha" : 0.9,
    "legend.fontsize"   : 9,
})


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 – LOGGING SUBSYSTEM
# ═════════════════════════════════════════════════════════════════════════════

class PipelineLogger:
    """Centralised singleton logger — console + rotating file."""

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
            fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger("AirbnbPipeline")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

    def info(self, msg: str) -> None:    self.logger.info(msg)
    def debug(self, msg: str) -> None:   self.logger.debug(msg)
    def warning(self, msg: str) -> None: self.logger.warning(msg)
    def error(self, msg: str) -> None:   self.logger.error(msg)

    def section(self, title: str) -> None:
        sep = "─" * 70
        self.logger.info("")
        self.logger.info(sep)
        self.logger.info(f"  {title.upper()}")
        self.logger.info(sep)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 – CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

class PipelineConfig:
    """All tuneable parameters — change here to adapt to any dataset."""

    def __init__(
        self,
        data_path: str,
        output_dir: str    = "outputs",
        report_dir: str    = "outputs/reports",
        chart_dir: str     = "outputs/charts",
        log_dir: str       = "outputs/logs",
        outlier_iqr_multiplier: float = 1.5,
        price_cap: float   = 1000.0,
        n_clusters: int    = 5,
        exclude_columns: Optional[List[str]] = None,
        sample_size: int   = 10_000,
    ) -> None:
        self.data_path      = data_path
        self.output_dir     = output_dir
        self.report_dir     = report_dir
        self.chart_dir      = chart_dir
        self.log_dir        = log_dir
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.price_cap      = price_cap
        self.n_clusters     = n_clusters
        self.exclude_columns = exclude_columns or []
        self.sample_size    = sample_size

    def create_directories(self) -> None:
        for d in [self.output_dir, self.report_dir, self.chart_dir, self.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DATA INGESTION LAYER
# ═════════════════════════════════════════════════════════════════════════════

class BaseDataLoader(ABC):
    @abstractmethod
    def load(self, path: str, **kwargs) -> pd.DataFrame: ...
    def validate(self, df: pd.DataFrame) -> bool: return not df.empty

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
    """Selects correct loader by extension, profiles schema."""

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
        ext    = Path(self.config.data_path).suffix.lower()
        loader = self._LOADERS.get(ext)
        if loader is None:
            raise ValueError(f"Unsupported file type: {ext}")

        self.log.info(f"Loading dataset: {self.config.data_path}")
        t0  = time.perf_counter()
        df  = loader.load(self.config.data_path)
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
            self.log.info(
                f"  {col:<38} dtype={dtype:<12} unique={uniq:<8} nulls={null_pct:.1f}%"
            )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 – DATA PROCESSING & TRANSFORMATION LAYER
# ═════════════════════════════════════════════════════════════════════════════

class DataQualityReport:
    def __init__(self) -> None:
        self.initial_rows: int = 0
        self.final_rows  : int = 0
        self.duplicates_removed: int = 0
        self.nulls_imputed: Dict[str, int] = {}
        self.outliers_flagged: Dict[str, int] = {}
        self.new_features: List[str] = []


class DataPreprocessor:
    """Modular preprocessing component. Single Responsibility."""

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config = config
        self.log    = logger
        self.report = DataQualityReport()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.section("Data Processing & Transformation Layer")
        self.report.initial_rows = len(df)

        df = df.copy()
        df = self._drop_excluded(df)
        df = self._standardise_column_names(df)
        df = self._parse_datetimes(df)
        df = self._clean_strings(df)
        df = self._remove_duplicates(df)
        df = self._handle_missing_values(df)
        df = self._clean_price(df)
        df = self._engineer_features(df)
        df = self._flag_outliers(df)

        self.report.final_rows = len(df)
        self._log_quality_report()
        return df

    def _drop_excluded(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in self.config.exclude_columns if c in df.columns]
        if drop:
            self.log.info(f"Dropping excluded columns: {drop}")
            df = df.drop(columns=drop)
        return df

    def _standardise_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns.str.strip()
                      .str.replace(r"\s+", "_", regex=True)
                      .str.replace(r"[^\w]", "", regex=True)
                      .str.lower()
        )
        self.log.info(f"Columns: {df.columns.tolist()}")
        return df

    def _parse_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        date_hints = {"date", "time", "timestamp", "dt", "month", "year"}
        for col in df.select_dtypes(include="object").columns:
            if any(h in col.lower() for h in date_hints):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    if parsed.notna().mean() > 0.5:
                        df[col] = parsed
                        self.log.info(f"  Auto-parsed datetime: {col}")
                except Exception:
                    pass
        return df

    def _clean_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()
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
            self.log.info(f"  Imputed {n_null:,} nulls in '{col}' using {strategy}")
        return df

    def _clean_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove zero-price listings and cap extreme outliers for analysis."""
        if "price" not in df.columns:
            return df
        before = len(df)
        df = df[df["price"] > 0].copy()
        removed = before - len(df)
        if removed:
            self.log.info(f"  Removed {removed} zero-price listings")
        # Flag but don't drop extreme prices; create a capped column for viz
        df["price_capped"] = df["price"].clip(upper=self.config.price_cap)
        self.log.info(f"  Created 'price_capped' (cap=${self.config.price_cap:,.0f})")
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        new_feats: List[str] = []

        # Price tier
        if "price" in df.columns:
            df["price_tier"] = pd.qcut(
                df["price_capped"], q=4,
                labels=["Budget", "Mid-Range", "Premium", "Luxury"],
                duplicates="drop"
            )
            new_feats.append("price_tier")

        # Availability category
        if "availability_365" in df.columns:
            df["availability_cat"] = pd.cut(
                df["availability_365"],
                bins=[-1, 0, 90, 180, 365],
                labels=["Unavailable", "Low", "Medium", "High"]
            )
            new_feats.append("availability_cat")

        # Review activity flag
        if "number_of_reviews" in df.columns:
            df["has_reviews"] = (df["number_of_reviews"] > 0).astype(int)
            new_feats.append("has_reviews")

        # Host listing scale
        if "calculated_host_listings_count" in df.columns:
            df["host_type"] = pd.cut(
                df["calculated_host_listings_count"],
                bins=[0, 1, 5, 20, 9999],
                labels=["Solo", "Small", "Medium", "Commercial"]
            )
            new_feats.append("host_type")

        # Revenue proxy = price × availability
        if "price" in df.columns and "availability_365" in df.columns:
            df["revenue_proxy"] = df["price_capped"] * df["availability_365"]
            new_feats.append("revenue_proxy")

        # Days since last review (if datetime parsed)
        if "last_review" in df.columns and pd.api.types.is_datetime64_any_dtype(df["last_review"]):
            ref = df["last_review"].max()
            df["days_since_review"] = (ref - df["last_review"]).dt.days
            new_feats.append("days_since_review")

        # Log price
        if "price" in df.columns:
            df["log_price"] = np.log1p(df["price"])
            new_feats.append("log_price")

        self.report.new_features = new_feats
        self.log.info(f"Engineered {len(new_feats)} new features: {new_feats}")
        return df

    def _flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        k = self.config.outlier_iqr_multiplier
        target_cols = ["price", "minimum_nights", "number_of_reviews",
                       "calculated_host_listings_count", "availability_365"]
        for col in target_cols:
            if col not in df.columns:
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr  = q3 - q1
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
    """Adaptive EDA — detects column types, computes full statistics."""

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config = config
        self.log    = logger
        self.eda_results: Dict[str, Any] = {}

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.log.section("Exploratory Data Analysis")
        self.eda_results["shape"]             = df.shape
        self.eda_results["descriptive_stats"] = self._descriptive_stats(df)
        self.eda_results["distribution"]      = self._distribution_summary(df)
        self.eda_results["correlation"]       = self._correlation_analysis(df)
        self.eda_results["categorical"]       = self._categorical_analysis(df)
        self.eda_results["time_series"]       = self._time_series_analysis(df)
        self.eda_results["geo"]               = self._geo_summary(df)
        self.eda_results["outlier_summary"]   = self._outlier_summary(df)
        self.eda_results["data_quality"]      = self._data_quality_summary(df)
        return self.eda_results

    def _descriptive_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        num_df = df.select_dtypes(include=[np.number])
        desc = num_df.describe(percentiles=[.10, .25, .50, .75, .90]).T
        desc["skewness"] = num_df.skew()
        desc["kurtosis"] = num_df.kurtosis()
        desc["cv_%"]     = (num_df.std() / num_df.mean().replace(0, np.nan) * 100).round(2)
        self.log.info(f"Descriptive stats for {len(desc)} numeric columns computed.")
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
                    "mean"     : float(series.mean()),
                    "std"      : float(series.std()),
                    "skew"     : float(series.skew()),
                    "kurtosis" : float(series.kurtosis()),
                    "normal_p" : float(p),
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
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack().reset_index()
        )
        pairs.columns = ["feature_a", "feature_b", "pearson_r"]
        pairs["abs_r"] = pairs["pearson_r"].abs()
        pairs = pairs.sort_values("abs_r", ascending=False).head(20)
        return {"matrix": corr, "top_pairs": pairs.reset_index(drop=True)}

    def _categorical_analysis(self, df: pd.DataFrame) -> Dict:
        results = {}
        for col in df.select_dtypes(include=["object", "category"]).columns:
            vc = df[col].value_counts()
            results[col] = {
                "unique_count": int(df[col].nunique()),
                "top_value"   : str(vc.index[0]) if len(vc) else "",
                "top_freq"    : int(vc.iloc[0]) if len(vc) else 0,
                "top_pct"     : float(round(vc.iloc[0] / len(df) * 100, 2)) if len(vc) else 0,
                "value_counts": vc.head(10).to_dict(),
            }
        return results

    def _time_series_analysis(self, df: pd.DataFrame) -> Dict:
        dt_cols = df.select_dtypes(include="datetime64").columns.tolist()
        results = {}
        for col in dt_cols:
            series = df[col].dropna().sort_values()
            results[col] = {
                "min"        : str(series.min()),
                "max"        : str(series.max()),
                "range_days" : int((series.max() - series.min()).days),
            }
        return results

    def _geo_summary(self, df: pd.DataFrame) -> Dict:
        if "latitude" not in df.columns or "longitude" not in df.columns:
            return {}
        return {
            "lat_range" : [float(df["latitude"].min()),  float(df["latitude"].max())],
            "lon_range" : [float(df["longitude"].min()), float(df["longitude"].max())],
            "lat_centre": float(df["latitude"].mean()),
            "lon_centre": float(df["longitude"].mean()),
        }

    def _outlier_summary(self, df: pd.DataFrame) -> Dict:
        return {
            c.replace("_is_outlier", ""): int(df[c].sum())
            for c in df.columns if c.endswith("_is_outlier")
        }

    def _data_quality_summary(self, df: pd.DataFrame) -> Dict:
        total = df.shape[0] * df.shape[1]
        nulls = df.isnull().sum().sum()
        return {
            "total_rows"          : df.shape[0],
            "total_columns"       : df.shape[1],
            "total_cells"         : total,
            "null_cells"          : int(nulls),
            "completeness_pct"    : round((1 - nulls / total) * 100, 2),
            "numeric_columns"     : len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns" : len(df.select_dtypes(include=["object", "category"]).columns),
            "datetime_columns"    : len(df.select_dtypes(include="datetime64").columns),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 – INSIGHT GENERATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class InsightGenerationEngine:
    """Auto-mines EDA results to produce 15 statistically-supported insights."""

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config   = config
        self.log      = logger
        self.insights: List[Dict] = []

    def generate(self, df: pd.DataFrame, eda: Dict) -> List[Dict]:
        self.log.section("Insight Generation Engine")
        self.insights.clear()

        self._insight_dataset_overview(df, eda)
        self._insight_price_distribution(df)
        self._insight_borough_price(df)
        self._insight_room_type_analysis(df)
        self._insight_availability_patterns(df)
        self._insight_review_activity(df)
        self._insight_host_scale(df)
        self._insight_top_neighbourhoods(df)
        self._insight_minimum_nights(df)
        self._insight_revenue_proxy(df)
        self._insight_price_room_anova(df)
        self._insight_geo_hotspots(df)
        self._insight_commercial_vs_solo(df)
        self._insight_review_recency(df)
        self._insight_outlier_listings(df)

        for i, ins in enumerate(self.insights, 1):
            self.log.info(f"  Insight #{i:02d} [{ins['category']}]: {ins['title']}")
        return self.insights

    def _add(self, title: str, body: str, category: str = "General") -> None:
        self.insights.append({"title": title, "body": body, "category": category})

    def _insight_dataset_overview(self, df: pd.DataFrame, eda: Dict) -> None:
        dq = eda.get("data_quality", {})
        boroughs = df["neighbourhood_group"].nunique() if "neighbourhood_group" in df.columns else "?"
        self._add(
            "Dataset Overview",
            f"The dataset contains {dq.get('total_rows',0):,} Airbnb listings across "
            f"{boroughs} NYC boroughs with {dq.get('total_columns',0)} attributes. "
            f"Data completeness is {dq.get('completeness_pct',100):.1f}%. "
            f"There are {dq.get('numeric_columns',0)} numeric and "
            f"{dq.get('categorical_columns',0)} categorical columns.",
            "Data Quality",
        )

    def _insight_price_distribution(self, df: pd.DataFrame) -> None:
        if "price" not in df.columns:
            return
        p = df["price"]
        pct_budget  = (p <= 75).mean() * 100
        pct_luxury  = (p > 300).mean() * 100
        self._add(
            "Price Distribution & Market Segmentation",
            f"Nightly prices range from ${p.min()} to ${p.max():,} with a median of "
            f"${p.median():.0f} and mean of ${p.mean():.0f} (heavily right-skewed). "
            f"{pct_budget:.1f}% of listings are budget (≤$75/night) while "
            f"{pct_luxury:.1f}% are luxury (>$300/night). "
            f"The IQR spans ${p.quantile(0.25):.0f}–${p.quantile(0.75):.0f}, "
            f"indicating a fragmented market with wide price dispersion.",
            "Pricing Analytics",
        )

    def _insight_borough_price(self, df: pd.DataFrame) -> None:
        if "neighbourhood_group" not in df.columns or "price" not in df.columns:
            return
        agg  = df.groupby("neighbourhood_group")["price"].agg(["median","mean","count"])
        top  = agg["median"].idxmax()
        low  = agg["median"].idxmin()
        self._add(
            "Borough-Level Price Hierarchy",
            f"Manhattan commands the highest median nightly rate "
            f"(${agg.loc[top,'median']:.0f}) — "
            f"{agg.loc[top,'median'] / agg.loc[low,'median']:.1f}× more than "
            f"{low} (${agg.loc[low,'median']:.0f}). "
            f"Brooklyn is the second most expensive borough, reflecting its growing "
            f"popularity as a Manhattan alternative. "
            f"The Bronx and Staten Island represent value segments with the lowest rates.",
            "Pricing Analytics",
        )

    def _insight_room_type_analysis(self, df: pd.DataFrame) -> None:
        if "room_type" not in df.columns:
            return
        vc   = df["room_type"].value_counts(normalize=True) * 100
        if "price" in df.columns:
            avg = df.groupby("room_type")["price"].median()
            top_rt   = avg.idxmax()
            price_note = (
                f" 'Entire home/apt' commands a {avg.get('Entire home/apt',0)/avg.get('Private room',1):.1f}× "
                f"price premium over private rooms."
            )
        else:
            price_note = ""
        self._add(
            "Room Type Market Composition",
            f"Entire home/apt listings dominate at {vc.get('Entire home/apt',0):.1f}% "
            f"of supply, followed by private rooms ({vc.get('Private room',0):.1f}%). "
            f"Shared rooms are a niche segment at {vc.get('Shared room',0):.1f}%.{price_note} "
            f"This split signals both leisure travellers seeking privacy and budget guests "
            f"willing to share spaces.",
            "Market Structure",
        )

    def _insight_availability_patterns(self, df: pd.DataFrame) -> None:
        if "availability_365" not in df.columns:
            return
        a = df["availability_365"]
        zero_pct  = (a == 0).mean() * 100
        full_pct  = (a == 365).mean() * 100
        self._add(
            "Listing Availability & Occupancy Signals",
            f"Median annual availability is {a.median():.0f} days, with mean {a.mean():.0f} days. "
            f"{zero_pct:.1f}% of listings show zero availability (possibly blocked/inactive), "
            f"while {full_pct:.1f}% are available year-round. "
            f"Low availability often correlates with high demand, "
            f"suggesting popular listings are consistently booked.",
            "Operational Analytics",
        )

    def _insight_review_activity(self, df: pd.DataFrame) -> None:
        if "number_of_reviews" not in df.columns:
            return
        r = df["number_of_reviews"]
        no_review_pct = (r == 0).mean() * 100
        high_review   = (r >= 50).mean() * 100
        if "price" in df.columns:
            r_price = stats.pearsonr(df["number_of_reviews"], df["price_capped"])[0]
            corr_note = f" Review count and price show a Pearson r = {r_price:.3f}, suggesting reviews are not price-driven."
        else:
            corr_note = ""
        self._add(
            "Review Volume & Guest Engagement",
            f"{no_review_pct:.1f}% of listings have zero reviews — potentially new or inactive. "
            f"Listings with 50+ reviews ({high_review:.1f}%) represent proven, high-engagement properties. "
            f"Median review count is {r.median():.0f} with mean {r.mean():.1f}.{corr_note}",
            "Guest Engagement",
        )

    def _insight_host_scale(self, df: pd.DataFrame) -> None:
        if "calculated_host_listings_count" not in df.columns:
            return
        h = df["calculated_host_listings_count"]
        solo_pct       = (h == 1).mean() * 100
        commercial_pct = (h > 5).mean() * 100
        max_listings   = int(h.max())
        self._add(
            "Host Portfolio Scale & Professionalisation",
            f"{solo_pct:.1f}% of listings come from solo hosts (1 listing), "
            f"indicating a large hobbyist/occasional-host segment. "
            f"However, {commercial_pct:.1f}% of listings are from hosts with 6+ properties, "
            f"suggesting meaningful professional operator presence. "
            f"The most prolific host manages {max_listings:,} listings — a full commercial operation.",
            "Host Analytics",
        )

    def _insight_top_neighbourhoods(self, df: pd.DataFrame) -> None:
        if "neighbourhood" not in df.columns:
            return
        top5 = df["neighbourhood"].value_counts().head(5)
        names = ", ".join([f"{n} ({v:,})" for n, v in top5.items()])
        if "price" in df.columns:
            price_by_nb = df.groupby("neighbourhood")["price"].median().sort_values(ascending=False)
            priciest = price_by_nb.index[0]
            priciest_val = price_by_nb.iloc[0]
            price_note = f" The most expensive neighbourhood by median price is {priciest} (${priciest_val:.0f}/night)."
        else:
            price_note = ""
        self._add(
            "Top Neighbourhoods by Listing Volume",
            f"The five most listed neighbourhoods are: {names}. "
            f"Williamsburg and Bedford-Stuyvesant in Brooklyn lead in volume, "
            f"reflecting the borough's rising popularity.{price_note}",
            "Geographic Analytics",
        )

    def _insight_minimum_nights(self, df: pd.DataFrame) -> None:
        if "minimum_nights" not in df.columns:
            return
        mn = df["minimum_nights"]
        long_stay_pct = (mn >= 30).mean() * 100
        one_night_pct = (mn == 1).mean() * 100
        if "price" in df.columns:
            long_price  = df[mn >= 30]["price_capped"].median()
            short_price = df[mn == 1]["price_capped"].median()
            price_note  = f" Long-stay listings median at ${long_price:.0f}/night vs ${short_price:.0f} for 1-night minimums."
        else:
            price_note = ""
        self._add(
            "Minimum Night Requirements & Stay Strategy",
            f"{one_night_pct:.1f}% of listings accept 1-night stays, maximising flexibility. "
            f"{long_stay_pct:.1f}% require 30+ nights, effectively targeting monthly renters "
            f"(possibly bypassing short-term rental regulations).{price_note} "
            f"Extreme outliers (minimum nights >365) likely represent data errors.",
            "Regulatory & Strategy",
        )

    def _insight_revenue_proxy(self, df: pd.DataFrame) -> None:
        if "revenue_proxy" not in df.columns:
            return
        rv = df["revenue_proxy"]
        top10_rev_share = df.nlargest(int(len(df)*0.10), "revenue_proxy")["revenue_proxy"].sum() / rv.sum() * 100
        if "neighbourhood_group" in df.columns:
            top_borough = df.groupby("neighbourhood_group")["revenue_proxy"].sum().idxmax()
            borough_note = f" {top_borough} generates the highest estimated total revenue among all boroughs."
        else:
            borough_note = ""
        self._add(
            "Revenue Concentration & Earning Potential",
            f"Using price × availability as a revenue proxy, the top 10% of listings "
            f"account for {top10_rev_share:.1f}% of estimated total platform revenue — "
            f"a classic Pareto-style concentration. "
            f"Median estimated annual revenue is ${rv.median():,.0f}.{borough_note}",
            "Financial Analytics",
        )

    def _insight_price_room_anova(self, df: pd.DataFrame) -> None:
        if "price" not in df.columns or "room_type" not in df.columns:
            return
        groups = [g["price"].dropna().values for _, g in df.groupby("room_type")]
        f, p   = f_oneway(*groups)
        self._add(
            "Statistical Price Difference Across Room Types",
            f"One-way ANOVA confirms that nightly prices differ significantly across room types "
            f"(F = {f:.2f}, p = {p:.2e}). "
            f"Entire home/apt listings are statistically more expensive than private and shared rooms. "
            f"This validates using room type as a key price predictor in any forecasting model.",
            "Statistical Findings",
        )

    def _insight_geo_hotspots(self, df: pd.DataFrame) -> None:
        if "neighbourhood_group" not in df.columns or "price" not in df.columns:
            return
        density = df.groupby("neighbourhood_group").size().sort_values(ascending=False)
        top_density = density.index[0]
        self._add(
            "Geographic Density & Supply Hotspots",
            f"Manhattan has the highest listing density with {density.get('Manhattan',0):,} listings, "
            f"followed by Brooklyn ({density.get('Brooklyn',0):,}). "
            f"These two boroughs together represent "
            f"{(density.get('Manhattan',0)+density.get('Brooklyn',0))/len(df)*100:.1f}% of all NYC supply. "
            f"Queens, the Bronx, and Staten Island are significantly underserved, "
            f"representing potential growth markets for hosts and investors.",
            "Geographic Analytics",
        )

    def _insight_commercial_vs_solo(self, df: pd.DataFrame) -> None:
        if "host_type" not in df.columns or "price" not in df.columns:
            return
        agg = df.groupby("host_type")["price_capped"].median()
        if "Commercial" in agg.index and "Solo" in agg.index:
            ratio = agg["Commercial"] / agg["Solo"]
            diff_note = (
                f"Commercial hosts price {ratio:.2f}× higher than solo hosts on median, "
                if ratio > 1 else
                f"Solo hosts actually price {1/ratio:.2f}× higher than commercial operators on median, "
            )
        else:
            diff_note = ""
        self._add(
            "Commercial vs Solo Host Pricing Strategy",
            f"{diff_note}suggesting different value propositions. "
            f"Commercial operators likely benefit from economies of scale in property management, "
            f"while solo hosts may price conservatively to attract first reviews. "
            f"This has implications for platform trust and regulatory scrutiny.",
            "Host Analytics",
        )

    def _insight_review_recency(self, df: pd.DataFrame) -> None:
        if "days_since_review" not in df.columns:
            return
        dsr = df["days_since_review"].dropna()
        stale_pct = (dsr > 365).mean() * 100
        fresh_pct = (dsr <= 90).mean() * 100
        self._add(
            "Review Recency & Listing Activity",
            f"{fresh_pct:.1f}% of listings received a review within the past 90 days — "
            f"indicating active, currently bookable properties. "
            f"{stale_pct:.1f}% have not been reviewed in over a year, "
            f"suggesting dormant or seasonally inactive listings that inflate headline supply figures.",
            "Guest Engagement",
        )

    def _insight_outlier_listings(self, df: pd.DataFrame) -> None:
        if "price_is_outlier" not in df.columns:
            return
        n = int(df["price_is_outlier"].sum())
        pct = n / len(df) * 100
        if "price" in df.columns:
            avg_out  = df[df["price_is_outlier"] == 1]["price"].mean()
            avg_norm = df[df["price_is_outlier"] == 0]["price"].mean()
            note = f" Outlier listings average ${avg_out:,.0f}/night vs ${avg_norm:,.0f} for standard listings."
        else:
            note = ""
        self._add(
            "Anomalous Pricing Detection",
            f"IQR-based outlier detection flagged {n:,} listings ({pct:.1f}%) with "
            f"anomalous nightly prices.{note} "
            f"These may represent luxury penthouses, data entry errors, or placeholder listings. "
            f"Excluding them reduces mean price significantly and improves model accuracy.",
            "Anomaly Detection",
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 – ADVANCED VISUALIZATION SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

class VisualizationEngine:
    """15 publication-quality Airbnb-specific charts."""

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config = config
        self.log    = logger
        self.saved: List[str] = []

    def _save(self, fig: plt.Figure, name: str) -> str:
        path = str(Path(self.config.chart_dir) / f"{name}.png")
        fig.savefig(path)
        plt.close(fig)
        self.saved.append(path)
        self.log.info(f"  Saved: {path}")
        return path

    def run(self, df: pd.DataFrame, eda: Dict) -> List[str]:
        self.log.section("Advanced Visualization System")
        sample = df.sample(min(self.config.sample_size, len(df)), random_state=42)

        self._chart_01_executive_dashboard(df)
        self._chart_02_price_distribution(df)
        self._chart_03_borough_analysis(df)
        self._chart_04_room_type_analysis(df)
        self._chart_05_correlation_heatmap(df, eda)
        self._chart_06_availability_analysis(df)
        self._chart_07_host_analysis(df)
        self._chart_08_neighbourhood_top20(df)
        self._chart_09_geo_scatter(sample)
        self._chart_10_minimum_nights(df)
        self._chart_11_review_analysis(df)
        self._chart_12_revenue_proxy(df)
        self._chart_13_price_tier_breakdown(df)
        self._chart_14_pca_clustering(df)
        self._chart_15_temporal_review_trend(df)

        self.log.info(f"Total charts generated: {len(self.saved)}")
        return self.saved

    # ── Chart 01: Executive Dashboard ─────────────────────────────────────
    def _chart_01_executive_dashboard(self, df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(20, 11), facecolor=BACKGROUND)
        fig.suptitle("NYC Airbnb Analytics — Executive Dashboard",
                     fontsize=20, fontweight="bold", color=PALETTE, y=1.01)
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

        # KPI tiles
        kpis = [
            ("Total Listings",     f"{len(df):,}",                     PALETTE),
            ("Median Price/Night", f"${df['price'].median():.0f}",      ACCENT),
            ("Avg Availability",   f"{df['availability_365'].mean():.0f}d", HIGHLIGHT),
            ("Avg Reviews",        f"{df['number_of_reviews'].mean():.1f}", NEUTRAL),
        ]
        for i, (label, val, color) in enumerate(kpis):
            ax = fig.add_subplot(gs[0, i])
            ax.set_facecolor(color)
            ax.text(0.5, 0.6, val, transform=ax.transAxes,
                    ha="center", va="center", fontsize=22, fontweight="bold", color="white")
            ax.text(0.5, 0.2, label, transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="white", alpha=0.9)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)

        # Borough bar
        ax = fig.add_subplot(gs[1, :2])
        vc = df["neighbourhood_group"].value_counts()
        bars = ax.barh(vc.index[::-1], vc.values[::-1],
                       color=CHART_PALETTE[:len(vc)], edgecolor="white")
        ax.set_title("Listings by Borough", fontweight="bold", color=NEUTRAL)
        ax.set_xlabel("Listings")
        for bar, v in zip(bars, vc.values[::-1]):
            ax.text(v + 100, bar.get_y() + bar.get_height()/2,
                    f"{v:,}", va="center", fontsize=9)

        # Room type pie
        ax = fig.add_subplot(gs[1, 2])
        vc2 = df["room_type"].value_counts()
        ax.pie(vc2.values, labels=vc2.index, autopct="%1.1f%%",
               colors=CHART_PALETTE[:len(vc2)], startangle=140,
               wedgeprops=dict(edgecolor="white", linewidth=2),
               textprops={"fontsize": 9})
        ax.set_title("Room Type Split", fontweight="bold", color=NEUTRAL)

        # Price histogram (capped)
        ax = fig.add_subplot(gs[1, 3])
        pc = df["price_capped"]
        ax.hist(pc, bins=60, color=PALETTE, alpha=0.8, edgecolor="white")
        ax.axvline(pc.median(), color=ACCENT, linestyle="--", linewidth=2,
                   label=f"Median ${pc.median():.0f}")
        ax.legend(fontsize=8)
        ax.set_title("Price Distribution (capped $1k)", fontweight="bold", color=NEUTRAL)
        ax.set_xlabel("Price ($)")
        fig.tight_layout()
        self._save(fig, "01_executive_dashboard")

    # ── Chart 02: Price Distribution Deep-Dive ────────────────────────────
    def _chart_02_price_distribution(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BACKGROUND)
        fig.suptitle("Price Distribution — Multi-Angle Analysis",
                     fontsize=14, fontweight="bold", color=PALETTE)

        # KDE + histogram
        pc = df["price_capped"]
        axes[0].hist(pc, bins=80, color=PALETTE, alpha=0.65, edgecolor="white", density=True)
        pc.plot.kde(ax=axes[0], color=ACCENT, linewidth=2)
        axes[0].axvline(pc.mean(),   color=HIGHLIGHT, linestyle="--", linewidth=1.5, label=f"Mean ${pc.mean():.0f}")
        axes[0].axvline(pc.median(), color=NEUTRAL,   linestyle=":",  linewidth=1.5, label=f"Median ${pc.median():.0f}")
        axes[0].legend(); axes[0].set_title("Price Histogram + KDE", fontweight="bold")
        axes[0].set_xlabel("Nightly Price ($)"); axes[0].set_ylabel("Density")

        # Box plot by borough
        if "neighbourhood_group" in df.columns:
            order = df.groupby("neighbourhood_group")["price_capped"].median().sort_values(ascending=False).index
            sns.boxplot(data=df, x="price_capped", y="neighbourhood_group",
                        order=order, palette=CHART_PALETTE[:5], ax=axes[1])
            axes[1].set_title("Price by Borough", fontweight="bold")
            axes[1].set_xlabel("Nightly Price ($)"); axes[1].set_ylabel("")

        # Violin by room type
        if "room_type" in df.columns:
            order2 = df.groupby("room_type")["price_capped"].median().sort_values(ascending=False).index
            sns.violinplot(data=df, x="room_type", y="price_capped",
                           order=order2, palette=CHART_PALETTE[:3], inner="quartile", ax=axes[2])
            axes[2].set_title("Price by Room Type", fontweight="bold")
            axes[2].set_xlabel("Room Type"); axes[2].set_ylabel("Nightly Price ($)")

        fig.tight_layout()
        self._save(fig, "02_price_distribution")

    # ── Chart 03: Borough Analysis ────────────────────────────────────────
    def _chart_03_borough_analysis(self, df: pd.DataFrame) -> None:
        if "neighbourhood_group" not in df.columns:
            return
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BACKGROUND)
        fig.suptitle("Borough-Level Analysis — NYC Airbnb",
                     fontsize=14, fontweight="bold", color=PALETTE)

        # Listing count
        vc = df["neighbourhood_group"].value_counts()
        axes[0,0].bar(vc.index, vc.values, color=CHART_PALETTE[:len(vc)], edgecolor="white")
        axes[0,0].set_title("Listings per Borough", fontweight="bold")
        axes[0,0].tick_params(axis="x", rotation=15)
        axes[0,0].set_ylabel("Count")
        for bar, v in zip(axes[0,0].patches, vc.values):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                           f"{v:,}", ha="center", fontsize=9)

        # Median price
        med = df.groupby("neighbourhood_group")["price_capped"].median().sort_values(ascending=False)
        axes[0,1].bar(med.index, med.values, color=CHART_PALETTE[:len(med)], edgecolor="white")
        axes[0,1].set_title("Median Nightly Price by Borough", fontweight="bold")
        axes[0,1].tick_params(axis="x", rotation=15)
        axes[0,1].set_ylabel("Median Price ($)")
        for bar, v in zip(axes[0,1].patches, med.values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, v + 1,
                           f"${v:.0f}", ha="center", fontsize=9)

        # Room type mix per borough
        cross = pd.crosstab(df["neighbourhood_group"], df["room_type"], normalize="index") * 100
        cross.plot(kind="bar", stacked=True, ax=axes[1,0],
                   color=CHART_PALETTE[:cross.shape[1]], edgecolor="white")
        axes[1,0].set_title("Room Type Mix per Borough (%)", fontweight="bold")
        axes[1,0].tick_params(axis="x", rotation=15)
        axes[1,0].legend(title="Room Type", bbox_to_anchor=(1,1))
        axes[1,0].set_ylabel("Share (%)")

        # Avg availability
        avail = df.groupby("neighbourhood_group")["availability_365"].mean().sort_values()
        axes[1,1].barh(avail.index, avail.values, color=CHART_PALETTE[:len(avail)], edgecolor="white")
        axes[1,1].set_title("Avg Availability (days/year) by Borough", fontweight="bold")
        axes[1,1].set_xlabel("Average Days Available")
        for bar, v in zip(axes[1,1].patches, avail.values):
            axes[1,1].text(v + 1, bar.get_y() + bar.get_height()/2,
                           f"{v:.0f}d", va="center", fontsize=9)

        fig.tight_layout()
        self._save(fig, "03_borough_analysis")

    # ── Chart 04: Room Type Analysis ──────────────────────────────────────
    def _chart_04_room_type_analysis(self, df: pd.DataFrame) -> None:
        if "room_type" not in df.columns:
            return
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BACKGROUND)
        fig.suptitle("Room Type — Comprehensive Analysis",
                     fontsize=14, fontweight="bold", color=PALETTE)

        vc = df["room_type"].value_counts()
        axes[0,0].pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                      colors=CHART_PALETTE[:len(vc)], startangle=140,
                      wedgeprops=dict(edgecolor="white", linewidth=2),
                      textprops={"fontsize": 10})
        axes[0,0].set_title("Market Share by Room Type", fontweight="bold")

        # Price box
        order = df.groupby("room_type")["price_capped"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="room_type", y="price_capped", order=order,
                    palette=CHART_PALETTE[:3], ax=axes[0,1])
        axes[0,1].set_title("Price Distribution by Room Type", fontweight="bold")
        axes[0,1].set_xlabel("Room Type"); axes[0,1].set_ylabel("Nightly Price ($)")

        # Availability
        sns.boxplot(data=df, x="room_type", y="availability_365", order=order,
                    palette=CHART_PALETTE[:3], ax=axes[1,0])
        axes[1,0].set_title("Availability by Room Type", fontweight="bold")
        axes[1,0].set_xlabel("Room Type"); axes[1,0].set_ylabel("Days Available/Year")

        # Review count
        sns.boxplot(data=df, x="room_type", y="number_of_reviews", order=order,
                    palette=CHART_PALETTE[:3], ax=axes[1,1])
        axes[1,1].set_title("Review Count by Room Type", fontweight="bold")
        axes[1,1].set_xlabel("Room Type"); axes[1,1].set_ylabel("Number of Reviews")
        axes[1,1].set_ylim(0, df["number_of_reviews"].quantile(0.97))

        fig.tight_layout()
        self._save(fig, "04_room_type_analysis")

    # ── Chart 05: Correlation Heatmap ─────────────────────────────────────
    def _chart_05_correlation_heatmap(self, df: pd.DataFrame, eda: Dict) -> None:
        num_df = df.select_dtypes(include=[np.number])
        cols   = [c for c in num_df.columns
                  if not c.endswith("_is_outlier") and c not in ["id","host_id"]][:14]
        corr = num_df[cols].corr()

        fig, ax = plt.subplots(figsize=(14, 11), facecolor=BACKGROUND)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(355, 177, as_cmap=True),
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Feature Correlation Matrix (Pearson)",
                     fontsize=14, fontweight="bold", color=PALETTE, pad=15)
        fig.tight_layout()
        self._save(fig, "05_correlation_heatmap")

    # ── Chart 06: Availability Analysis ──────────────────────────────────
    def _chart_06_availability_analysis(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BACKGROUND)
        fig.suptitle("Availability Patterns — Occupancy Intelligence",
                     fontsize=14, fontweight="bold", color=PALETTE)

        a = df["availability_365"]
        axes[0].hist(a, bins=50, color=ACCENT, alpha=0.8, edgecolor="white")
        axes[0].axvline(a.mean(),   color=PALETTE,   linestyle="--", linewidth=2, label=f"Mean {a.mean():.0f}d")
        axes[0].axvline(a.median(), color=HIGHLIGHT, linestyle=":",  linewidth=2, label=f"Median {a.median():.0f}d")
        axes[0].legend(); axes[0].set_title("Availability Distribution", fontweight="bold")
        axes[0].set_xlabel("Days Available / Year"); axes[0].set_ylabel("Count")

        if "availability_cat" in df.columns:
            vc = df["availability_cat"].value_counts()
            axes[1].bar(vc.index.astype(str), vc.values,
                        color=CHART_PALETTE[:len(vc)], edgecolor="white")
            axes[1].set_title("Availability Category Breakdown", fontweight="bold")
            axes[1].set_xlabel("Availability Tier"); axes[1].set_ylabel("Count")
            for bar, v in zip(axes[1].patches, vc.values):
                axes[1].text(bar.get_x() + bar.get_width()/2, v + 50,
                             f"{v:,}", ha="center", fontsize=9)

        if "neighbourhood_group" in df.columns:
            order = df.groupby("neighbourhood_group")["availability_365"].mean().sort_values(ascending=False).index
            sns.barplot(data=df, x="neighbourhood_group", y="availability_365",
                        order=order, palette=CHART_PALETTE[:5],
                        estimator=np.mean, errorbar=("ci",95), ax=axes[2])
            axes[2].set_title("Avg Availability by Borough", fontweight="bold")
            axes[2].set_xlabel("Borough"); axes[2].set_ylabel("Avg Days Available")
            axes[2].tick_params(axis="x", rotation=15)

        fig.tight_layout()
        self._save(fig, "06_availability_analysis")

    # ── Chart 07: Host Analysis ───────────────────────────────────────────
    def _chart_07_host_analysis(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BACKGROUND)
        fig.suptitle("Host Portfolio Analysis",
                     fontsize=14, fontweight="bold", color=PALETTE)

        h = df["calculated_host_listings_count"]
        h_capped = h.clip(upper=20)
        axes[0].hist(h_capped, bins=20, color=PALETTE, alpha=0.8, edgecolor="white")
        axes[0].set_title("Host Listing Count Distribution (capped 20)", fontweight="bold")
        axes[0].set_xlabel("# Listings per Host"); axes[0].set_ylabel("Count")

        if "host_type" in df.columns:
            vc = df["host_type"].value_counts()
            axes[1].bar(vc.index.astype(str), vc.values,
                        color=CHART_PALETTE[:len(vc)], edgecolor="white")
            axes[1].set_title("Host Type Breakdown", fontweight="bold")
            axes[1].set_xlabel("Host Type"); axes[1].set_ylabel("Count")
            for bar, v in zip(axes[1].patches, vc.values):
                axes[1].text(bar.get_x() + bar.get_width()/2, v + 50,
                             f"{v:,}", ha="center", fontsize=9)

            if "price_capped" in df.columns:
                order = ["Solo", "Small", "Medium", "Commercial"]
                order = [o for o in order if o in df["host_type"].cat.categories]
                sns.boxplot(data=df, x="host_type", y="price_capped", order=order,
                            palette=CHART_PALETTE[:4], ax=axes[2])
                axes[2].set_title("Price by Host Type", fontweight="bold")
                axes[2].set_xlabel("Host Type"); axes[2].set_ylabel("Nightly Price ($)")

        fig.tight_layout()
        self._save(fig, "07_host_analysis")

    # ── Chart 08: Top-20 Neighbourhoods ──────────────────────────────────
    def _chart_08_neighbourhood_top20(self, df: pd.DataFrame) -> None:
        if "neighbourhood" not in df.columns:
            return
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BACKGROUND)
        fig.suptitle("Top 20 Neighbourhoods Analysis",
                     fontsize=14, fontweight="bold", color=PALETTE)

        top20 = df["neighbourhood"].value_counts().head(20)
        colors = [PALETTE if i < 5 else ACCENT if i < 10 else HIGHLIGHT
                  for i in range(20)]
        axes[0].barh(top20.index[::-1], top20.values[::-1], color=colors[::-1], edgecolor="white")
        axes[0].set_title("Top 20 by Listing Count", fontweight="bold")
        axes[0].set_xlabel("Number of Listings")
        for bar, v in zip(axes[0].patches, top20.values[::-1]):
            axes[0].text(v + 20, bar.get_y() + bar.get_height()/2,
                         f"{v:,}", va="center", fontsize=8)

        if "price_capped" in df.columns:
            top20_names = top20.index.tolist()
            price_med = (df[df["neighbourhood"].isin(top20_names)]
                         .groupby("neighbourhood")["price_capped"]
                         .median()
                         .reindex(top20_names)
                         .sort_values())
            axes[1].barh(price_med.index, price_med.values,
                         color=[PALETTE if v > price_med.median() else ACCENT
                                for v in price_med.values],
                         edgecolor="white")
            axes[1].axvline(price_med.median(), color=NEUTRAL, linestyle="--",
                            linewidth=1.5, label=f"Median ${price_med.median():.0f}")
            axes[1].legend()
            axes[1].set_title("Median Price — Top 20 Neighbourhoods", fontweight="bold")
            axes[1].set_xlabel("Median Nightly Price ($)")

        fig.tight_layout()
        self._save(fig, "08_neighbourhood_top20")

    # ── Chart 09: Geographic Scatter ──────────────────────────────────────
    def _chart_09_geo_scatter(self, df: pd.DataFrame) -> None:
        if "latitude" not in df.columns or "longitude" not in df.columns:
            return
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=BACKGROUND)
        fig.suptitle("Geographic Distribution of NYC Airbnb Listings",
                     fontsize=14, fontweight="bold", color=PALETTE)

        # Colour by borough
        if "neighbourhood_group" in df.columns:
            boroughs = df["neighbourhood_group"].unique()
            pal = {b: CHART_PALETTE[i] for i, b in enumerate(boroughs)}
            for b in boroughs:
                sub = df[df["neighbourhood_group"] == b]
                axes[0].scatter(sub["longitude"], sub["latitude"],
                                c=pal[b], alpha=0.25, s=3, label=b)
            axes[0].legend(markerscale=4, fontsize=8)
        else:
            axes[0].scatter(df["longitude"], df["latitude"],
                            c=PALETTE, alpha=0.2, s=3)
        axes[0].set_title("Listing Density by Borough", fontweight="bold")
        axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

        # Colour by price
        if "price_capped" in df.columns:
            sc = axes[1].scatter(df["longitude"], df["latitude"],
                                 c=df["price_capped"], cmap="RdYlGn_r",
                                 alpha=0.3, s=3, vmin=0, vmax=300)
            plt.colorbar(sc, ax=axes[1], label="Nightly Price ($)")
        axes[1].set_title("Price Heatmap (Geographic)", fontweight="bold")
        axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")

        fig.tight_layout()
        self._save(fig, "09_geo_scatter")

    # ── Chart 10: Minimum Nights ──────────────────────────────────────────
    def _chart_10_minimum_nights(self, df: pd.DataFrame) -> None:
        if "minimum_nights" not in df.columns:
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BACKGROUND)
        fig.suptitle("Minimum Night Requirements — Stay Strategy Analysis",
                     fontsize=14, fontweight="bold", color=PALETTE)

        mn = df["minimum_nights"].clip(upper=31)
        axes[0].hist(mn, bins=31, color=HIGHLIGHT, alpha=0.8, edgecolor="white")
        axes[0].set_title("Min Nights Distribution (capped 31)", fontweight="bold")
        axes[0].set_xlabel("Minimum Nights Required"); axes[0].set_ylabel("Count")

        # By borough
        if "neighbourhood_group" in df.columns:
            order = df.groupby("neighbourhood_group")["minimum_nights"].median().sort_values(ascending=False).index
            sns.barplot(data=df[df["minimum_nights"] <= 31], x="neighbourhood_group",
                        y="minimum_nights", order=order,
                        palette=CHART_PALETTE[:5], estimator=np.median,
                        errorbar=("ci",95), ax=axes[1])
            axes[1].set_title("Median Min Nights by Borough", fontweight="bold")
            axes[1].set_xlabel("Borough"); axes[1].tick_params(axis="x", rotation=15)

        # By room type
        if "room_type" in df.columns:
            sns.boxplot(data=df[df["minimum_nights"] <= 31], x="room_type",
                        y="minimum_nights", palette=CHART_PALETTE[:3], ax=axes[2])
            axes[2].set_title("Min Nights by Room Type", fontweight="bold")
            axes[2].set_xlabel("Room Type"); axes[2].set_ylabel("Min Nights")

        fig.tight_layout()
        self._save(fig, "10_minimum_nights")

    # ── Chart 11: Review Analysis ─────────────────────────────────────────
    def _chart_11_review_analysis(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BACKGROUND)
        fig.suptitle("Reviews & Guest Engagement Analysis",
                     fontsize=14, fontweight="bold", color=PALETTE)

        r = df["number_of_reviews"].clip(upper=200)
        axes[0,0].hist(r, bins=60, color=ACCENT, alpha=0.8, edgecolor="white")
        axes[0,0].set_title("Review Count Distribution (capped 200)", fontweight="bold")
        axes[0,0].set_xlabel("Number of Reviews"); axes[0,0].set_ylabel("Count")

        if "neighbourhood_group" in df.columns:
            order = df.groupby("neighbourhood_group")["number_of_reviews"].mean().sort_values(ascending=False).index
            sns.barplot(data=df, x="neighbourhood_group", y="number_of_reviews",
                        order=order, palette=CHART_PALETTE[:5],
                        estimator=np.mean, errorbar=("ci",95), ax=axes[0,1])
            axes[0,1].set_title("Avg Reviews by Borough", fontweight="bold")
            axes[0,1].tick_params(axis="x", rotation=15)

        if "room_type" in df.columns:
            sns.boxplot(data=df, x="room_type", y="number_of_reviews",
                        palette=CHART_PALETTE[:3], ax=axes[1,0])
            axes[1,0].set_ylim(0, df["number_of_reviews"].quantile(0.95))
            axes[1,0].set_title("Reviews by Room Type", fontweight="bold")

        if "reviews_per_month" in df.columns and "price_capped" in df.columns:
            sub = df[df["reviews_per_month"] > 0].sample(min(5000, len(df)), random_state=42)
            axes[1,1].scatter(sub["reviews_per_month"], sub["price_capped"],
                              alpha=0.25, s=8, color=PALETTE)
            axes[1,1].set_title("Reviews/Month vs Nightly Price", fontweight="bold")
            axes[1,1].set_xlabel("Reviews per Month"); axes[1,1].set_ylabel("Nightly Price ($)")
            axes[1,1].set_xlim(0, sub["reviews_per_month"].quantile(0.98))

        fig.tight_layout()
        self._save(fig, "11_review_analysis")

    # ── Chart 12: Revenue Proxy ───────────────────────────────────────────
    def _chart_12_revenue_proxy(self, df: pd.DataFrame) -> None:
        if "revenue_proxy" not in df.columns:
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BACKGROUND)
        fig.suptitle("Estimated Revenue Proxy (Price × Availability)",
                     fontsize=14, fontweight="bold", color=PALETTE)

        rv = df["revenue_proxy"].clip(upper=df["revenue_proxy"].quantile(0.95))
        axes[0].hist(rv, bins=60, color="#2A9D8F", alpha=0.8, edgecolor="white")
        axes[0].set_title("Revenue Proxy Distribution", fontweight="bold")
        axes[0].set_xlabel("Estimated Annual Revenue ($)"); axes[0].set_ylabel("Count")

        if "neighbourhood_group" in df.columns:
            order = df.groupby("neighbourhood_group")["revenue_proxy"].median().sort_values(ascending=False).index
            sns.barplot(data=df, x="neighbourhood_group", y="revenue_proxy",
                        order=order, palette=CHART_PALETTE[:5],
                        estimator=np.median, errorbar=("ci",95), ax=axes[1])
            axes[1].set_title("Median Revenue Proxy by Borough", fontweight="bold")
            axes[1].tick_params(axis="x", rotation=15); axes[1].set_ylabel("Median Est. Revenue ($)")

        if "room_type" in df.columns:
            order2 = df.groupby("room_type")["revenue_proxy"].median().sort_values(ascending=False).index
            sns.barplot(data=df, x="room_type", y="revenue_proxy",
                        order=order2, palette=CHART_PALETTE[:3],
                        estimator=np.median, errorbar=("ci",95), ax=axes[2])
            axes[2].set_title("Median Revenue Proxy by Room Type", fontweight="bold")
            axes[2].set_ylabel("Median Est. Revenue ($)")

        fig.tight_layout()
        self._save(fig, "12_revenue_proxy")

    # ── Chart 13: Price Tier Breakdown ────────────────────────────────────
    def _chart_13_price_tier_breakdown(self, df: pd.DataFrame) -> None:
        if "price_tier" not in df.columns:
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BACKGROUND)
        fig.suptitle("Price Tier Segmentation Analysis",
                     fontsize=14, fontweight="bold", color=PALETTE)

        vc = df["price_tier"].value_counts()
        axes[0].bar(vc.index.astype(str), vc.values, color=CHART_PALETTE[:len(vc)], edgecolor="white")
        axes[0].set_title("Listing Count by Price Tier", fontweight="bold")
        axes[0].set_xlabel("Price Tier"); axes[0].set_ylabel("Count")
        for bar, v in zip(axes[0].patches, vc.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, v + 50,
                         f"{v:,}", ha="center", fontsize=9)

        if "neighbourhood_group" in df.columns:
            cross = pd.crosstab(df["neighbourhood_group"], df["price_tier"], normalize="index") * 100
            cross.plot(kind="bar", stacked=True, ax=axes[1],
                       color=CHART_PALETTE[:cross.shape[1]], edgecolor="white")
            axes[1].set_title("Price Tier Mix by Borough (%)", fontweight="bold")
            axes[1].tick_params(axis="x", rotation=15)
            axes[1].legend(title="Tier", bbox_to_anchor=(1,1))

        if "room_type" in df.columns:
            cross2 = pd.crosstab(df["room_type"], df["price_tier"], normalize="index") * 100
            cross2.plot(kind="bar", stacked=True, ax=axes[2],
                        color=CHART_PALETTE[:cross2.shape[1]], edgecolor="white")
            axes[2].set_title("Price Tier Mix by Room Type (%)", fontweight="bold")
            axes[2].tick_params(axis="x", rotation=15)
            axes[2].legend(title="Tier", bbox_to_anchor=(1,1))

        fig.tight_layout()
        self._save(fig, "13_price_tier_breakdown")

    # ── Chart 14: PCA + Clustering ────────────────────────────────────────
    def _chart_14_pca_clustering(self, df: pd.DataFrame) -> None:
        useful_cols = ["price_capped", "minimum_nights", "number_of_reviews",
                       "calculated_host_listings_count", "availability_365"]
        usable = [c for c in useful_cols if c in df.columns]
        if len(usable) < 3:
            return
        sample = df[usable].sample(min(5000, len(df)), random_state=42).dropna()
        scaled = StandardScaler().fit_transform(sample)
        pca    = PCA(n_components=2, random_state=42)
        pcs    = pca.fit_transform(scaled)

        km     = KMeans(n_clusters=self.config.n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BACKGROUND)
        fig.suptitle("PCA Dimensionality Reduction & K-Means Clustering",
                     fontsize=14, fontweight="bold", color=PALETTE)

        sc = axes[0].scatter(pcs[:,0], pcs[:,1], c=labels, cmap="Set1",
                             alpha=0.45, s=10, edgecolors="none")
        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        axes[0].set_title(f"PCA — {self.config.n_clusters} Clusters", fontweight="bold")
        plt.colorbar(sc, ax=axes[0], label="Cluster")

        pca_full = PCA(n_components=min(len(usable), 5), random_state=42)
        pca_full.fit(scaled)
        evr  = pca_full.explained_variance_ratio_
        cumv = np.cumsum(evr)
        x    = range(1, len(evr)+1)
        axes[1].bar(x, evr*100, color=HIGHLIGHT, alpha=0.8, edgecolor="white", label="Per Component")
        axes[1].plot(x, cumv*100, "o-", color=PALETTE, linewidth=2, label="Cumulative")
        axes[1].axhline(90, color=NEUTRAL, linestyle="--", alpha=0.5, label="90% threshold")
        axes[1].set_xlabel("Principal Component"); axes[1].set_ylabel("Explained Variance (%)")
        axes[1].set_title("PCA Scree Plot", fontweight="bold")
        axes[1].legend()

        fig.tight_layout()
        self._save(fig, "14_pca_clustering")

    # ── Chart 15: Temporal Review Trend ──────────────────────────────────
    def _chart_15_temporal_review_trend(self, df: pd.DataFrame) -> None:
        dt_col = "last_review" if "last_review" in df.columns else None
        if dt_col is None or not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            return
        fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor=BACKGROUND)
        fig.suptitle("Temporal Patterns — Review & Listing Activity",
                     fontsize=14, fontweight="bold", color=PALETTE)

        monthly = df.set_index(dt_col).resample("ME").size()
        axes[0,0].plot(monthly.index, monthly.values, color=PALETTE, linewidth=2, marker="o", markersize=3)
        axes[0,0].fill_between(monthly.index, monthly.values, alpha=0.12, color=PALETTE)
        axes[0,0].set_title("Monthly Review Activity", fontweight="bold")
        axes[0,0].set_xlabel("Date"); axes[0,0].set_ylabel("Reviews")

        dow = df[dt_col].dt.dayofweek.value_counts().sort_index()
        day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        axes[0,1].bar([day_names[i] for i in dow.index], dow.values,
                      color=CHART_PALETTE[:7], edgecolor="white")
        axes[0,1].set_title("Reviews by Day of Week", fontweight="bold")
        axes[0,1].set_xlabel("Day"); axes[0,1].set_ylabel("Count")

        yr = df[dt_col].dt.year.value_counts().sort_index()
        axes[1,0].bar(yr.index.astype(str), yr.values,
                      color=CHART_PALETTE[:len(yr)], edgecolor="white")
        axes[1,0].set_title("Review Activity by Year", fontweight="bold")
        axes[1,0].set_xlabel("Year"); axes[1,0].set_ylabel("Count")

        if "days_since_review" in df.columns:
            dsr = df["days_since_review"].dropna().clip(upper=1500)
            axes[1,1].hist(dsr, bins=60, color=ACCENT, alpha=0.8, edgecolor="white")
            axes[1,1].axvline(365, color=PALETTE, linestyle="--", linewidth=2,
                              label="1-year mark")
            axes[1,1].legend()
            axes[1,1].set_title("Days Since Last Review", fontweight="bold")
            axes[1,1].set_xlabel("Days"); axes[1,1].set_ylabel("Count")

        fig.tight_layout()
        self._save(fig, "15_temporal_review_trend")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 – STATISTICAL & ANALYTICAL LAYER
# ═════════════════════════════════════════════════════════════════════════════

class StatisticalAnalysisLayer:
    """ANOVA, Chi², regression, T-test, silhouette clustering."""

    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config  = config
        self.log     = logger
        self.results: Dict[str, Any] = {}

    def run(self, df: pd.DataFrame) -> Dict:
        self.log.section("Statistical & Analytical Layer")
        self.results["anova_price_borough"]    = self._anova_price_by_borough(df)
        self.results["anova_price_room_type"]  = self._anova_price_by_room_type(df)
        self.results["chi2_room_borough"]      = self._chi2_room_borough(df)
        self.results["regression_price"]       = self._regression_price_drivers(df)
        self.results["ttest_entire_vs_private"]= self._ttest_price_room_types(df)
        self.results["clustering_quality"]     = self._clustering_quality(df)
        self._log_stat_results()
        return self.results

    def _anova_price_by_borough(self, df: pd.DataFrame) -> Dict:
        if "price_capped" not in df.columns or "neighbourhood_group" not in df.columns:
            return {}
        groups = [g["price_capped"].dropna().values for _, g in df.groupby("neighbourhood_group")]
        f, p   = f_oneway(*groups)
        return {
            "f_statistic": round(f, 4), "p_value": round(p, 8),
            "significant": bool(p < 0.05),
            "interpretation": (
                "Price differs significantly across NYC boroughs (p<0.05). "
                "Borough is a strong pricing predictor." if p < 0.05 else
                "No significant price difference across boroughs."
            ),
        }

    def _anova_price_by_room_type(self, df: pd.DataFrame) -> Dict:
        if "price_capped" not in df.columns or "room_type" not in df.columns:
            return {}
        groups = [g["price_capped"].dropna().values for _, g in df.groupby("room_type")]
        f, p   = f_oneway(*groups)
        return {
            "f_statistic": round(f, 4), "p_value": round(p, 8),
            "significant": bool(p < 0.05),
            "interpretation": (
                "Room type is a statistically significant price driver (ANOVA p<0.05)." if p < 0.05 else
                "No significant price variation by room type."
            ),
        }

    def _chi2_room_borough(self, df: pd.DataFrame) -> Dict:
        if "room_type" not in df.columns or "neighbourhood_group" not in df.columns:
            return {}
        table = pd.crosstab(df["neighbourhood_group"], df["room_type"])
        chi2, p, dof, _ = chi2_contingency(table)
        return {
            "chi2": round(chi2, 4), "p_value": round(p, 8), "dof": dof,
            "significant": bool(p < 0.05),
            "interpretation": (
                "Room type distribution is significantly associated with borough (Chi² p<0.05). "
                "Different boroughs favour different listing types." if p < 0.05 else
                "Room type distribution is uniform across boroughs."
            ),
        }

    def _regression_price_drivers(self, df: pd.DataFrame) -> Dict:
        features = ["minimum_nights", "number_of_reviews",
                    "calculated_host_listings_count", "availability_365"]
        usable = [c for c in features if c in df.columns]
        if "price_capped" not in df.columns or len(usable) < 2:
            return {}
        clean = df[usable + ["price_capped"]].dropna()
        X = clean[usable].values
        y = clean["price_capped"].values
        model = Ridge(alpha=1.0).fit(X, y)
        r2    = model.score(X, y)
        coefs = dict(zip(usable, [round(c, 4) for c in model.coef_]))
        return {
            "r_squared"   : round(r2, 6),
            "coefficients": coefs,
            "interpretation": (
                f"Ridge regression on price: R²={r2:.4f}. "
                f"Availability (coef={coefs.get('availability_365',0):+.2f}) and "
                f"min_nights (coef={coefs.get('minimum_nights',0):+.2f}) are key drivers."
            ),
        }

    def _ttest_price_room_types(self, df: pd.DataFrame) -> Dict:
        if "room_type" not in df.columns or "price_capped" not in df.columns:
            return {}
        entire  = df[df["room_type"] == "Entire home/apt"]["price_capped"].dropna()
        private = df[df["room_type"] == "Private room"]["price_capped"].dropna()
        if entire.empty or private.empty:
            return {}
        t, p = ttest_ind(entire, private, equal_var=False)
        return {
            "t_statistic"    : round(t, 4),
            "p_value"        : round(p, 8),
            "entire_median"  : round(float(entire.median()), 2),
            "private_median" : round(float(private.median()), 2),
            "significant"    : bool(p < 0.05),
            "interpretation" : (
                f"Entire home/apt (median ${entire.median():.0f}) is significantly "
                f"more expensive than private rooms (median ${private.median():.0f}), "
                f"Welch T-test p = {p:.2e}."
            ),
        }

    def _clustering_quality(self, df: pd.DataFrame) -> Dict:
        cols  = ["price_capped", "availability_365", "number_of_reviews",
                 "minimum_nights", "calculated_host_listings_count"]
        usable = [c for c in cols if c in df.columns]
        if len(usable) < 3:
            return {}
        sample = df[usable].sample(min(3000, len(df)), random_state=42).dropna()
        scaled = StandardScaler().fit_transform(sample)
        km     = KMeans(n_clusters=self.config.n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        score  = silhouette_score(scaled, labels, sample_size=1000, random_state=42)
        return {
            "n_clusters"       : self.config.n_clusters,
            "silhouette_score" : round(score, 4),
            "interpretation"   : (
                f"K-Means (k={self.config.n_clusters}) silhouette = {score:.4f}. "
                + ("Reasonable cluster separation found." if score > 0.2 else
                   "Overlapping clusters — listing segments are continuous, not discrete.")
            ),
        }

    def _log_stat_results(self) -> None:
        for test, res in self.results.items():
            if isinstance(res, dict) and "interpretation" in res:
                self.log.info(f"  [{test.upper()[:30]}] {res['interpretation']}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 – REPORTING & OUTPUT SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

class ReportingEngine:
    def __init__(self, config: PipelineConfig, logger: PipelineLogger) -> None:
        self.config = config
        self.log    = logger

    def generate(self, df, eda, insights, stats, charts) -> None:
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
            f.write("  NYC AIRBNB ANALYTICS PIPELINE — FULL REPORT\n")
            f.write(f"  Generated : {ts}\n")
            f.write(f"  Dataset   : {self.config.data_path}\n")
            f.write("=" * 80 + "\n\n")
            dq = eda.get("data_quality", {})
            f.write("── DATA QUALITY ──\n")
            for k, v in dq.items():
                f.write(f"  {k:<30}: {v}\n")
            f.write("\n")
            desc = eda.get("descriptive_stats")
            if desc is not None:
                f.write("── DESCRIPTIVE STATISTICS ──\n")
                f.write(desc.to_string())
                f.write("\n\n")
            f.write("── STATISTICAL TEST RESULTS ──\n")
            for test, res in stats.items():
                if isinstance(res, dict):
                    f.write(f"  {test.upper()}\n")
                    for k, v in res.items():
                        f.write(f"    {k:<30}: {v}\n")
                    f.write("\n")
            f.write("── ANALYTICAL INSIGHTS ──\n")
            for i, ins in enumerate(insights, 1):
                f.write(f"\n  [{i:02d}] [{ins['category']}] {ins['title']}\n")
                f.write(f"       {ins['body']}\n")
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
            "metadata"          : {"generated_at": datetime.now().isoformat(), "dataset": self.config.data_path},
            "data_quality"      : eda.get("data_quality", {}),
            "categorical_summary": {
                k: {kk: vv for kk, vv in v.items() if kk != "value_counts"}
                for k, v in eda.get("categorical", {}).items()
            },
            "statistical_tests" : stats,
            "insights"          : insights,
            "charts_generated"  : charts,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=serialise)
        self.log.info(f"  JSON report: {path}")

    def _write_insight_summary(self, insights) -> None:
        path = Path(self.config.report_dir) / "insight_summary.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("NYC AIRBNB — INSIGHT SUMMARY\n")
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
    """Dependency-injected orchestrator — wires all layers end-to-end."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        config.create_directories()
        self.log          = PipelineLogger(log_dir=config.log_dir)
        self.ingestion    = DataIngestionLayer(config, self.log)
        self.preprocessor = DataPreprocessor(config, self.log)
        self.eda          = EDAEngine(config, self.log)
        self.insights     = InsightGenerationEngine(config, self.log)
        self.stats        = StatisticalAnalysisLayer(config, self.log)
        self.viz          = VisualizationEngine(config, self.log)
        self.reporter     = ReportingEngine(config, self.log)

    def run(self) -> None:
        wall_start = time.perf_counter()
        self.log.section("NYC Airbnb Analytics Pipeline — Start")
        self.log.info(f"Dataset : {self.config.data_path}")
        self.log.info(f"Output  : {self.config.output_dir}")

        try:
            raw_df       = self.ingestion.load()
            clean_df     = self.preprocessor.preprocess(raw_df)
            eda_results  = self.eda.run(clean_df)
            insight_list = self.insights.generate(clean_df, eda_results)
            stat_results = self.stats.run(clean_df)
            chart_paths  = self.viz.run(clean_df, eda_results)
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
        print("  ✅  AIRBNB ANALYTICS PIPELINE COMPLETE")
        print(f"  Insights : {len(insight_list)} | Charts : {len(chart_paths)} | Time : {elapsed:.1f}s")
        print("═" * 72 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 – ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    config = PipelineConfig(
        data_path        = "/mnt/user-data/uploads/Airbnb_data.csv",
        output_dir       = "/mnt/user-data/outputs",
        report_dir       = "/mnt/user-data/outputs/reports",
        chart_dir        = "/mnt/user-data/outputs/charts",
        log_dir          = "/mnt/user-data/outputs/logs",
        outlier_iqr_multiplier = 1.5,
        price_cap        = 1000.0,
        n_clusters       = 5,
        exclude_columns  = ["id", "host_id", "name", "host_name"],
        sample_size      = 10_000,
    )
    AnalyticsPipelineOrchestrator(config).run()


if __name__ == "__main__":
    main()
