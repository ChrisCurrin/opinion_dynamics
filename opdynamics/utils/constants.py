"""Store constants"""
from opdynamics.utils.latex_helpers import math_fix

TIME_SYMBOL = "$t$"
OPINION_AGENT_SYMBOL = "$x_i$"
OPINION_AGENT_TIME = "$x_i (t)$"
P_A_X = "$P(a,x)$"
OPINION_SYMBOL = "$x$"
ACTIVITY_SYMBOL = "$a$"
ABS_MEAN_FINAL_OPINION = math_fix(f"$|\langle {OPINION_SYMBOL}_{{f}} \\rangle|$")
MEAN_NEAREST_NEIGHBOUR = math_fix(f"$\langle {OPINION_SYMBOL} \\rangle^{{NN}}$")

# has less contrast than seismic and the middle is more grey
OPINIONS_CMAP = "coolwarm_r"
#
INTERACTIONS_CMAP = "magma_r"

INTERNAL_NOISE = 10
EXTERNAL_NOISE = 20
