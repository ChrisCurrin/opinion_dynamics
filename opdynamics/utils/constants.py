"""Store constants"""
from opdynamics.utils.latex_helpers import math_fix

# ----------
# system values
# ----------
# noise
INTERNAL_NOISE_K = 10
INTERNAL_NOISE_CONTRAST = 12
EXTERNAL_NOISE = 0

# ----------
# other values
# ----------
DEFAULT_COMPRESSION_LEVEL = 7

# ----------
# colors
# ----------
# has less contrast than seismic and the middle is more grey
OPINIONS_CMAP = "coolwarm_r"
INTERACTIONS_CMAP = "Purples"
CONNECTIONS_CMAP = "YlGn"
LIGHT_BLUE = "#1ACAD9"
LIGHT_RED = "#D9258E"
LIGHT_GREEN = "#0FD977"
TAN = "#BB9832"

PRE_RDN_COLOR = LIGHT_RED
POST_RDN_COLOR = LIGHT_BLUE
POST_RECOVERY_COLOR = TAN
DELTA_RDN_COLOR = LIGHT_GREEN

# ----------
# symbols
# ----------
TIME_SYMBOL = "$t$"
OPINION_SYMBOL = "$x$"
OPINION_AGENT_SYMBOL = math_fix(f"${OPINION_SYMBOL}_i$")
OPINION_AGENT_TIME = math_fix(f"${OPINION_AGENT_SYMBOL} (t)$")
P_A_X = f"P($a$,{OPINION_SYMBOL})"
ACTIVITY_SYMBOL = "$a$"
ABS_MEAN_FINAL_OPINION = math_fix(f"$|\langle {OPINION_SYMBOL}_{{f}} \\rangle|$")
MEAN_NEAREST_NEIGHBOUR = math_fix(f"$\langle {OPINION_SYMBOL}^{{NN}} \\rangle$")
SAMPLE_MEAN = "$\overline{X}_n$"
MEAN = MU = "$\mu$"
PEAK_DISTANCE = math_fix(f"$\Lambda_{OPINION_SYMBOL}$")
PEAK_DISTANCE_MEAN = math_fix(f"$\\langle {PEAK_DISTANCE} \\rangle$")
PEAK_DISTANCE_VAR = math_fix(f"$\\sigma^2_{{{PEAK_DISTANCE}}}$")
IN_DEGREE_SYMBOL = "$→ i$"
OUT_DEGREE_SYMBOL = "$i →$"