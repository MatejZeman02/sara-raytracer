"""Ad hoc script to generate the graph for the RTX 4070 Ti Wavefront data, without the baseline comparison."""

import matplotlib.pyplot as plt

# Embeddeded RTX 4070 Ti Wavefront Data
budgets = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4000,
    5000,
    6000,
    7000,
    8000,
    10000,
    16000,
    32000,
]

p1_times = [
    0.0029,
    0.0030,
    0.0033,
    0.0041,
    0.0060,
    0.0099,
    0.0173,
    0.0302,
    0.0512,
    0.0608,
    0.0688,
    0.0759,
    0.0817,
    0.0914,
    0.1058,
    0.1098,
]

comp_times = [
    0.0163,
    0.0163,
    0.0178,
    0.0190,
    0.0214,
    0.0209,
    0.0196,
    0.0148,
    0.0117,
    0.0107,
    0.0086,
    0.0079,
    0.0069,
    0.0060,
    0.0038,
    0.0030,
]

p2_times = [
    0.1293,
    0.1288,
    0.1342,
    0.1362,
    0.1322,
    0.1410,
    0.1367,
    0.1139,
    0.0878,
    0.0727,
    0.0584,
    0.0456,
    0.0370,
    0.0270,
    0.0126,
    0.0040,
]

tot_times = [
    0.1484,
    0.1479,
    0.1546,
    0.1587,
    0.1593,
    0.1713,
    0.1737,
    0.1588,
    0.1512,
    0.1443,
    0.1363,
    0.1296,
    0.1251,
    0.1241,
    0.1222,
    0.1166,
]
fig, ax = plt.subplots(figsize=(12, 7))

# The Stacked Areas
ax.stackplot(
    budgets,
    p1_times,
    comp_times,
    p2_times,
    labels=[
        "Pass 1",
        "CPU Compaction",
        "Pass 2 (Cleanup)",
    ],
    colors=["#a1c9f4", "#ffb482", "#8de5a1"],
    alpha=0.8,
)

# Total Render Time Line
ax.plot(
    budgets,
    tot_times,
    color="black",
    linewidth=2.5,
    marker="o",
    label="Total Wavefront Time",
)

# Formatting
ax.set_title("Wavefront Stream Compaction Tradeoffs (RTX 4070 Ti)", fontsize=14, pad=15)
ax.set_xlabel("BVH Operations Budget (Pause Point)", fontsize=12)
ax.set_ylabel("Total Render Time (Seconds)", fontsize=12)

# Log scale for X axis to spread the early low budgets
ax.set_xscale("log")
ax.set_xticks(budgets)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
plt.xticks(rotation=45)  # Rotate labels so they don't overlap

ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=10)
plt.tight_layout()

out_png = "wavefront_4070.png"
plt.savefig(out_png, dpi=300)
print(f"Graph successfully saved to: {out_png}")
plt.show()
