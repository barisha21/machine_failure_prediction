import os
import matplotlib.pyplot as plt
import seaborn as sns


def save_all_plots(df):

    # Create Folder
    os.makedirs("artifacts/plots", exist_ok=True)

    # =====================================================
    # 1. Distribution of Numerical Features
    # =====================================================

    df.hist(figsize=(12, 8), bins=20)

    plt.suptitle("Distribution of Numerical Features")

    plt.savefig(
        "artifacts/plots/distribution_of_numerical_features.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # 2. Distribution of Machine Types
    # =====================================================

    plt.figure(figsize=(6,4))

    sns.countplot(
        x="Type",
        data=df
    )

    plt.title("Distribution of Machine Types")
    plt.xlabel("Machine Type")
    plt.ylabel("Count")

    plt.savefig(
        "artifacts/plots/distribution_of_machine_types.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # 3. Machine Failure Distribution
    # =====================================================

    failure = df["Machine failure"].value_counts()

    plt.figure(figsize=(6,6))

    plt.pie(
        failure,
        labels=["No Failure", "Failure"],
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Machine Failure Distribution")

    plt.savefig(
        "artifacts/plots/machine_failure_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # 4. Detection of Outliers
    # =====================================================

    plt.figure(figsize=(12,6))

    sns.boxplot(
        data=df[
            [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]
        ]
    )

    plt.title("Detection of Outliers in Numerical Features")

    plt.xticks(rotation=20)

    plt.savefig(
        "artifacts/plots/outlier_detection.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
        # =====================================================
    # 5. Correlation Between Numerical Features
    # =====================================================

    plt.figure(figsize=(8,6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Between Numerical Features")

    plt.savefig(
        "artifacts/plots/correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # 6. Relationship Between Torque and Rotational Speed
    # =====================================================

    plt.figure(figsize=(7,5))

    sns.scatterplot(
        x="Torque [Nm]",
        y="Rotational speed [rpm]",
        hue="Machine failure",
        data=df
    )

    plt.title("Relationship Between Torque and Rotational Speed")

    plt.savefig(
        "artifacts/plots/torque_vs_rotational_speed.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # 7. Trend of Tool Wear
    # =====================================================

    plt.figure(figsize=(10,5))

    plt.plot(
        df["Tool wear [min]"],
        linewidth=2
    )

    plt.title("Trend of Tool Wear")
    plt.xlabel("Observation")
    plt.ylabel("Tool Wear (min)")

    plt.savefig(
        "artifacts/plots/tool_wear_trend.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
        # =====================================================
    # 8. Distribution of Torque Based on Machine Failure
    # =====================================================

    plt.figure(figsize=(7,5))

    sns.violinplot(
        x="Machine failure",
        y="Torque [Nm]",
        data=df
    )

    plt.title("Distribution of Torque Based on Machine Failure")
    plt.xlabel("Machine Failure")
    plt.ylabel("Torque (Nm)")

    plt.savefig(
        "artifacts/plots/torque_distribution_based_on_machine_failure.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =====================================================
    # 9. Pairwise Relationship Between Numerical Features
    # =====================================================

    pair_plot = sns.pairplot(
        df[
            [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "Machine failure"
            ]
        ],
        hue="Machine failure"
    )

    pair_plot.fig.suptitle(
        "Pairwise Relationship Between Numerical Features",
        y=1.02
    )

    pair_plot.savefig(
        "artifacts/plots/pairwise_relationship_between_numerical_features.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =====================================================
    # 10. Average Torque Across Machine Types
    # =====================================================

    plt.figure(figsize=(7,5))

    sns.barplot(
        x="Type",
        y="Torque [Nm]",
        data=df
    )

    plt.title("Average Torque Across Machine Types")
    plt.xlabel("Machine Type")
    plt.ylabel("Average Torque (Nm)")

    plt.savefig(
        "artifacts/plots/average_torque_across_machine_types.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =====================================================
    # SUCCESS MESSAGE
    # =====================================================

    print("=" * 60)
    print("All EDA plots saved successfully!")
    print("Location : artifacts/plots/")
    print("=" * 60)