# tenspace: spacing and t-spacing tests for spiked tensor PCA
#
# Companion package of the article
#   Azais J.-M., Dalmao F., De Castro Y.,
#   "Second maximum of a Gaussian random field and exact (t-)spacing test",
#   arXiv:2406.18397
#
# Modules:
#   tenspace.pivot       p-values (direct and log-scale) and the Lemma-3 check
#   tenspace.simulation  spiked tensor model, gradient descents, Monte-Carlo
#   tenspace.plots       loading and plotting of the Monte-Carlo experiments
#                        (imports matplotlib/seaborn; import it explicitly)

from tenspace.pivot import (g_naive, h_naive, h_stable, is_valid_second_max,
                            log_g_stable, mills_ratio, spacing_pvalue,
                            tspacing_pvalue)
from tenspace.simulation import monte_carlo, one_replication

__version__ = "1.0.0"

__all__ = [
    "g_naive", "h_naive", "h_stable", "is_valid_second_max", "log_g_stable",
    "mills_ratio", "spacing_pvalue", "tspacing_pvalue",
    "monte_carlo", "one_replication",
]
