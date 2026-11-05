import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.config import PROJECT_ROOT, load_config

MONTH_ORDER = [
    "January", "February", "March",
    "April", "May", "June",
    "July", "August", "September",
    "October", "November", "December",
]


def save_figure(output_dir, filename):
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def main():
    config = load_config()
    df = pd.read_csv(PROJECT_ROOT / config["data_path"])

    output_dir = PROJECT_ROOT / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    print("=" * 60)
    print("STARTING EDA VISUALIZATIONS")
    print("=" * 60)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="IsCanceled")
    plt.title("Booking Cancellation Distribution")
    plt.xlabel("Is Canceled (0 = No, 1 = Yes)")
    plt.ylabel("Number of Bookings")
    save_figure(output_dir, "cancellation_distribution.png")

    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="LeadTime", bins=40, kde=True)
    plt.title("Lead Time Distribution")
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Number of Bookings")
    save_figure(output_dir, "lead_time_distribution.png")

    adr_for_plot = df[df["ADR"] <= 500]
    plt.figure(figsize=(9, 5))
    sns.histplot(data=adr_for_plot, x="ADR", bins=40, kde=True)
    plt.title("Average Daily Rate (ADR) Distribution")
    plt.xlabel("ADR")
    plt.ylabel("Number of Bookings")
    save_figure(output_dir, "adr_distribution.png")
    print("ADR values excluded from visualization only:", len(df) - len(adr_for_plot))

    monthly_bookings = df["ArrivalDateMonth"].value_counts().reindex(MONTH_ORDER)
    plt.figure(figsize=(11, 5))
    sns.barplot(x=monthly_bookings.index, y=monthly_bookings.values)
    plt.title("Number of Bookings Per Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Bookings")
    plt.xticks(rotation=45)
    save_figure(output_dir, "bookings_per_month.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="IsCanceled", y="LeadTime", showfliers=False)
    plt.title("Lead Time vs Booking Cancellation")
    plt.xlabel("Is Canceled (0 = No, 1 = Yes)")
    plt.ylabel("Lead Time (Days)")
    save_figure(output_dir, "lead_time_vs_cancellation.png")

    correlation_matrix = df.select_dtypes(include="number").corr()
    plt.figure(figsize=(15, 11))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap")
    save_figure(output_dir, "correlation_heatmap.png")

    print("\n" + "=" * 60)
    print("EDA VISUALIZATIONS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nSaved Figures:")
    for filename in sorted(p.name for p in output_dir.iterdir()):
        print(filename)


if __name__ == "__main__":
    main()
