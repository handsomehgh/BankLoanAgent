# author hgh
# version 1.0
"""
Real-time LPR data service
"""
import logging
from typing import Optional

import akshare
import pandas as pd

from config.models.bank_global_config import BankGlobalConfig
from infra.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class LPRDataService:
    """Get the latest 1-year and 5-year LPR"""

    CACHE_KEY = "latest_lpr"
    SUCCESS_TTL = 3600
    FAILURE_TTL = 60

    def __init__(self, config: BankGlobalConfig, cache: Optional[CacheManager] = None):
        self._cache = cache

        self._fallback_lpr = {
            "lpr_1y": config.lpr.lpr_1y,
            "lpr_5y": config.lpr.lpr_5y,
            "date": config.lpr.date,
            "source": "内置默认值（外部获取失败且缓存不可用）",
        }

    def get_latest_lpr(self) -> dict:
        """Obtain the latest LPR, prioritizing external real-time data, followed by cache, and finally default values"""
        # 1. real-time obtain
        try:
            lpr_data = self._fetch_from_akshare()
            if self._cache:
                self._cache.set(self.CACHE_KEY, lpr_data, ttl=self.SUCCESS_TTL)
            return lpr_data
        except Exception as e:
            logger.error("Failed to get LPR from AkShare: %s, trying to use cache", e)

        # 2. obtain from caceh
        if self._cache:
            cached = self._cache.get(self.CACHE_KEY)
            if cached and isinstance(cached, dict) and "lpr_1y" in cached:
                logger.warning("Using cached LPR data (may be outdated)")
                self._cache.set_null(self.CACHE_KEY, ttl=self.FAILURE_TTL)
                return cached

        # 3. default data
        logger.critical("All LPR data sources are unavailable, using built-in default values")
        return self._fallback_lpr

    def _fetch_from_akshare(self):
        """Get the latest LPR through AkShare"""
        lpr_df = akshare.macro_china_lpr()
        if lpr_df.empty:
            raise ValueError("AkShare empty LPR data")
        latest = lpr_df.sort_values('TRADE_DATE', ascending=False).iloc[0]
        return {
            "lpr_1y": float(latest['LPR1Y']),
            "lpr_5y": float(latest['LPR5Y']),
            "date": pd.Timestamp(latest['TRADE_DATE']).strftime('%Y-%m-%d'),
            "source": "AkShare (chinamoney.com.cn)",
        }
