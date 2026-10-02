import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load Dataset
# =========================

name = "train"
# name = "valid"
CSV_PATH = f"E:\\PROJECTS\\Alta2026\\Project\\data\\{name}.csv" 
OUTPUT_FOLDER = "E:\\PROJECTS\\Alta2026\\Project\\visualization\\result"

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


def visualize_distribution(df, column, figsize=(16, 6)):
    
    counts = df[column].value_counts(dropna=False)

    
    percentages = df[column].value_counts(
        normalize=True,
        dropna=False
    ) * 100

    labels = counts.index.astype(str)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    
    axes[0].pie(
        percentages.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    axes[0].set_title(
        f"{column.capitalize()} Distribution (Percentage)",
        fontsize=14,
        fontweight="bold"
    )

    bars = axes[1].bar(
        labels,
        percentages.values
    )

    axes[1].set_title(
        f"{column.capitalize()} Distribution (Percentage)",
        fontsize=14,
        fontweight="bold"
    )

    axes[1].set_xlabel(column.capitalize())
    axes[1].set_ylabel("Percentage (%)")
    axes[1].tick_params(axis="x", rotation=45)

    for bar, percentage in zip(bars, percentages.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.1f}%",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{OUTPUT_FOLDER}//{name}_{column}_distribution.png", dpi=300, bbox_inches='tight')
    

    print(f"\n{'=' * 50}")
    print(f"{column.upper()} DISTRIBUTION")
    print(f"{'=' * 50}")

    summary = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })

    print(summary)

def visualize_variety_sentiment_sarcasm(df, figsize=(18, 10)):
    """
    Visualize the relationship between variety, sentiment, and sarcasm.

    For each variety, calculate the percentage of all combinations:
        sentiment=0, sarcasm=0
        sentiment=0, sarcasm=1
        sentiment=1, sarcasm=0
        sentiment=1, sarcasm=1

    Then visualize them separately and compare all varieties.
    """

    # ==========================================
    # 1. Create a combined label for each sample
    # ==========================================

    data = df.copy()

    data["sentiment_sarcasm"] = (
        "Sentiment=" + data["sentiment"].astype(str)
        + " | Sarcasm=" + data["sarcasm"].astype(str)
    )

    # Define a consistent order
    combination_order = [
        "Sentiment=0 | Sarcasm=0",
        "Sentiment=0 | Sarcasm=1",
        "Sentiment=1 | Sarcasm=0",
        "Sentiment=1 | Sarcasm=1"
    ]

    # ==========================================
    # 2. Calculate percentages within each variety
    # ==========================================

    percentages = (
        pd.crosstab(
            data["variety"],
            data["sentiment_sarcasm"],
            normalize="index"
        )
        * 100
    )

    # Ensure all 4 combinations exist
    percentages = percentages.reindex(
        columns=combination_order,
        fill_value=0
    )

    # ==========================================
    # 3. Print detailed statistics
    # ==========================================

    counts = pd.crosstab(
        data["variety"],
        data["sentiment_sarcasm"]
    )

    counts = counts.reindex(
        columns=combination_order,
        fill_value=0
    )

    print("\n" + "=" * 80)
    print("VARIETY × SENTIMENT × SARCASM DISTRIBUTION")
    print("=" * 80)

    for variety in data["variety"].dropna().unique():

        print(f"\nVARIETY: {variety}")
        print("-" * 80)

        summary = pd.DataFrame({
            "Count": counts.loc[variety],
            "Percentage": percentages.loc[variety].round(2)
        })

        print(summary)

    # ==========================================
    # 4. Create comparison figure
    # ==========================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize
    )

    # ==========================================
    # Plot 1: Grouped Bar Chart
    # ==========================================

    percentages.T.plot(
        kind="bar",
        ax=axes[0]
    )

    axes[0].set_title(
        "Sentiment + Sarcasm Distribution by Variety",
        fontsize=15,
        fontweight="bold"
    )

    axes[0].set_xlabel(
        "Sentiment and Sarcasm Combination"
    )

    axes[0].set_ylabel(
        "Percentage (%)"
    )

    axes[0].tick_params(
        axis="x",
        rotation=30
    )

    axes[0].legend(
        title="Variety"
    )

    # Add percentage values above bars
    for container in axes[0].containers:
        axes[0].bar_label(
            container,
            fmt="%.1f%%",
            padding=3,
            fontsize=9
        )


    # ==========================================
    # Plot 2: Stacked Bar Chart
    # ==========================================

    percentages.plot(
        kind="bar",
        stacked=True,
        ax=axes[1]
    )

    axes[1].set_title(
        "Composition of Sentiment + Sarcasm in Each Variety",
        fontsize=15,
        fontweight="bold"
    )

    axes[1].set_xlabel(
        "Variety"
    )

    axes[1].set_ylabel(
        "Percentage (%)"
    )

    axes[1].set_ylim(0, 100)

    axes[1].tick_params(
        axis="x",
        rotation=0
    )

    axes[1].legend(
        title="Combination",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{OUTPUT_FOLDER}//{name}_Combination.png", dpi=300, bbox_inches='tight')



    # ==========================================
    # 5. Pie charts for each variety
    # ==========================================

    varieties = data["variety"].dropna().unique()

    fig, axes = plt.subplots(
        1,
        len(varieties),
        figsize=(7 * len(varieties), 6)
    )

    # Handle only one variety safely
    if len(varieties) == 1:
        axes = [axes]

    for ax, variety in zip(axes, varieties):

        values = percentages.loc[variety]

        ax.pie(
            values,
            labels=values.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title(
            f"{variety}\nSentiment + Sarcasm Distribution",
            fontsize=14,
            fontweight="bold"
        )

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{OUTPUT_FOLDER}//{name}_variety_sentiment_sarcasm.png", dpi=300, bbox_inches='tight')
    

def visualize_sentiment_and_sarcasm_by_variety(df, figsize=(16, 10)):
   
    varieties = df["variety"].dropna().unique()

    for variety in varieties:

        variety_df = df[df["variety"] == variety]


        sentiment_counts = variety_df["sentiment"].value_counts(
            dropna=False
        )

        sentiment_percentages = (
            sentiment_counts / sentiment_counts.sum()
        ) * 100


        sarcasm_counts = variety_df["sarcasm"].value_counts(
            dropna=False
        )

        sarcasm_percentages = (
            sarcasm_counts / sarcasm_counts.sum()
        ) * 100



        fig, axes = plt.subplots(
            2,
            2,
            figsize=figsize
        )

        fig.suptitle(
            f"Variety: {variety}",
            fontsize=18,
            fontweight="bold"
        )



        axes[0, 0].pie(
            sentiment_percentages.values,
            labels=sentiment_percentages.index.astype(str),
            autopct="%1.1f%%",
            startangle=90
        )

        axes[0, 0].set_title(
            "Sentiment Distribution",
            fontsize=14,
            fontweight="bold"
        )



        sentiment_bars = axes[0, 1].bar(
            sentiment_percentages.index.astype(str),
            sentiment_percentages.values
        )

        axes[0, 1].set_title(
            "Sentiment Distribution",
            fontsize=14,
            fontweight="bold"
        )

        axes[0, 1].set_xlabel("Sentiment")
        axes[0, 1].set_ylabel("Percentage (%)")

        for bar, percentage in zip(
            sentiment_bars,
            sentiment_percentages.values
        ):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{percentage:.1f}%",
                ha="center",
                va="bottom"
            )



        axes[1, 0].pie(
            sarcasm_percentages.values,
            labels=sarcasm_percentages.index.astype(str),
            autopct="%1.1f%%",
            startangle=90
        )

        axes[1, 0].set_title(
            "Sarcasm Distribution",
            fontsize=14,
            fontweight="bold"
        )


        sarcasm_bars = axes[1, 1].bar(
            sarcasm_percentages.index.astype(str),
            sarcasm_percentages.values
        )

        axes[1, 1].set_title(
            "Sarcasm Distribution",
            fontsize=14,
            fontweight="bold"
        )

        axes[1, 1].set_xlabel("Sarcasm")
        axes[1, 1].set_ylabel("Percentage (%)")

        for bar, percentage in zip(
            sarcasm_bars,
            sarcasm_percentages.values
        ):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{percentage:.1f}%",
                ha="center",
                va="bottom"
            )


        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{OUTPUT_FOLDER}//{name}_{variety}_sentiment_and_sarcasm_by_variety.png", dpi=300, bbox_inches='tight')


        

        print("\n" + "=" * 70)
        print(f"VARIETY: {variety}")
        print("=" * 70)

        print("\nSentiment Distribution:")

        sentiment_summary = pd.DataFrame({
            "Count": sentiment_counts,
            "Percentage": sentiment_percentages.round(2)
        })

        print(sentiment_summary)


        print("\nSarcasm Distribution:")

        sarcasm_summary = pd.DataFrame({
            "Count": sarcasm_counts,
            "Percentage": sarcasm_percentages.round(2)
        })

        print(sarcasm_summary)

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def analyze_categorical_relationship(
    df,
    column1,
    column2,
    alpha=0.05,
    correction=True
):
  

    contingency_table = pd.crosstab(
        df[column1],
        df[column2]
    )


    chi2, p_value, dof, expected = chi2_contingency(
        contingency_table,
        correction=correction
    )

    expected_df = pd.DataFrame(
        expected,
        index=contingency_table.index,
        columns=contingency_table.columns
    )


    n = contingency_table.to_numpy().sum()

    r, k = contingency_table.shape

    # Cramer's V
    denominator = n * min(r - 1, k - 1)

    cramers_v = np.sqrt(chi2 / denominator)

    

    if cramers_v < 0.1:
        effect_size = "Negligible / Very weak"
    elif cramers_v < 0.3:
        effect_size = "Weak"
    elif cramers_v < 0.5:
        effect_size = "Moderate"
    else:
        effect_size = "Strong"

   

    if p_value < alpha:
        conclusion = (
            f"Statistically significant relationship "
            f"(p < {alpha}). "
            f"We reject the null hypothesis of independence."
        )
    else:
        conclusion = (
            f"No statistically significant relationship "
            f"(p >= {alpha}). "
            f"We fail to reject the null hypothesis of independence."
        )

    print("\n" + "=" * 80)
    print(f"RELATIONSHIP ANALYSIS: {column1} × {column2}")
    print("=" * 80)

    print("\nContingency Table:")
    print(contingency_table)

    print("\nExpected Frequencies:")
    print(expected_df.round(2))

    print("\n" + "-" * 80)
    print("CHI-SQUARE TEST")
    print("-" * 80)

    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.10g}")
    print(f"Degrees of freedom: {dof}")

    print("\nConclusion:")
    print(conclusion)

    print("\n" + "-" * 80)
    print("CRAMER'S V")
    print("-" * 80)

    print(f"Cramer's V: {cramers_v:.4f}")
    print(f"Effect size: {effect_size}")


    low_expected_count = np.sum(expected < 5)
    total_cells = expected.size

    if low_expected_count > 0:
        print("\nWARNING:")
        print(
            f"{low_expected_count}/{total_cells} cells have "
            f"expected frequencies below 5."
        )
        print(
            "The Chi-square approximation may be less reliable."
        )


    return {
        "contingency_table": contingency_table,
        "chi2": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "expected_frequencies": expected_df,
        "cramers_v": cramers_v,
        "cramers_v_interpretation": effect_size,
        "conclusion": conclusion
    }

visualize_distribution(df, "variety")
visualize_distribution(df, "sarcasm")
visualize_distribution(df, "sentiment")
visualize_variety_sentiment_sarcasm(df)
visualize_sentiment_and_sarcasm_by_variety(df)
visualize_variety_sentiment_sarcasm(df)

# ==========================================
# 1. Variety × Sarcasm
# ==========================================

variety_sarcasm_results = analyze_categorical_relationship(
    df,
    "variety",
    "sarcasm"
)


# ==========================================
# 2. Variety × Sentiment
# ==========================================

variety_sentiment_results = analyze_categorical_relationship(
    df,
    "variety",
    "sentiment"
)

# ==========================================
# 3. Sentiment × Sarcasm
# ==========================================

sentiment_sarcasm_results = analyze_categorical_relationship(
    df,
    "sentiment",
    "sarcasm"
)