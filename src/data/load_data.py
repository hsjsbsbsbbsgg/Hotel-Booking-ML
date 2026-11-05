from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "H2.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_analysis"

EXPECTED_COLUMNS = [
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "AssignedRoomType",
    "BookingChanges",
    "DepositType",
    "Agent",
    "Company",
    "DaysInWaitingList",
    "CustomerType",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "ReservationStatus",
    "ReservationStatusDate",
]


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Make sure H2.csv is inside the project's data/ directory."
        )

    df = pd.read_csv(path)

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    if missing_columns:
        raise ValueError(
            "Dataset is missing expected source columns: "
            f"{missing_columns}"
        )

    if unexpected_columns:
        print("Warning: unexpected columns found:", unexpected_columns)

    return df


def analyze_dataset(df: pd.DataFrame) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    print_section("1. DATASET INFORMATION")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nColumn names:")
    print(df.columns.tolist())


    print_section("2. DATA TYPES")
    print(df.dtypes.to_string())


    print_section("3. MISSING VALUES")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing.to_string())
    print("\nTotal missing values:", int(missing.sum()))


    print_section("4. DUPLICATE ANALYSIS")
    duplicate_count = int(df.duplicated().sum())
    print("Total duplicate rows:", duplicate_count)
    print("\nFirst 10 duplicate records:")
    print(df[df.duplicated(keep=False)].head(10).to_string())


    print_section("5. STATISTICAL SUMMARY")
    summary = df.describe(include="all").T
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / "statistical_summary.csv")


    print_section("6. UNUSUAL VALUES")
    unusual_counts = {
        "Adults == 0": int((df["Adults"] == 0).sum()),
        "ADR > 500": int((df["ADR"] > 500).sum()),
        "Babies > 2": int((df["Babies"] > 2).sum()),
        "StaysInWeekNights > 10": int(
            (df["StaysInWeekNights"] > 10).sum()
        ),
        "DaysInWaitingList > 100": int(
            (df["DaysInWaitingList"] > 100).sum()
        ),
    }

    for description, count in unusual_counts.items():
        print(f"{description}: {count}")


    print_section("7. INVESTIGATION OF UNUSUAL RECORDS")

    unusual_records = {
        "adults_zero": df[df["Adults"] == 0],
        "adr_above_500": df[df["ADR"] > 500],
        "babies_above_2": df[df["Babies"] > 2],
    }

    for name, records in unusual_records.items():
        print(f"\n{name}:")
        print(records.head(10).to_string())


    for name, records in unusual_records.items():
        records.to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False,
        )


    print_section("8. ZERO GUEST ANALYSIS")
    children = df["Children"].fillna(0)
    zero_guests = df[
        (df["Adults"] == 0)
        & (children == 0)
        & (df["Babies"] == 0)
    ]
    print("Bookings with zero total guests:", len(zero_guests))
    zero_guests.to_csv(OUTPUT_DIR / "zero_guest_bookings.csv", index=False)


    print_section("9. COMPANY AND AGENT ANALYSIS")
    print("\nTop 10 Company values:")
    print(df["Company"].value_counts(dropna=False).head(10).to_string())
    print("\nTop 10 Agent values:")
    print(df["Agent"].value_counts(dropna=False).head(10).to_string())


    print_section("10. NULL VALUES IN AGENT AND COMPANY")
    agent_clean = df["Agent"].astype("string").str.strip()
    company_clean = df["Company"].astype("string").str.strip()

    agent_null_count = int(
        (agent_clean.str.upper() == "NULL").fillna(False).sum()
    )
    company_null_count = int(
        (company_clean.str.upper() == "NULL").fillna(False).sum()
    )

    print("Agent literal NULL:", agent_null_count)
    print("Company literal NULL:", company_null_count)


    print_section("11. CATEGORICAL CARDINALITY")
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    cardinality = df[categorical_columns].nunique(dropna=False).sort_values(
        ascending=False
    )
    print(cardinality.to_string())
    cardinality.to_csv(OUTPUT_DIR / "categorical_cardinality.csv")


    print_section("12. TARGET DISTRIBUTION")
    target_counts = df["IsCanceled"].value_counts(dropna=False).sort_index()
    print(target_counts.to_string())
    target_counts.to_csv(OUTPUT_DIR / "is_canceled_distribution.csv")


    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": duplicate_count,
        "agent_literal_null": agent_null_count,
        "company_literal_null": company_null_count,
        **unusual_counts,
    }

    pd.Series(report, name="value").to_csv(
        OUTPUT_DIR / "inspection_report.csv",
        header=True,
    )

    print_section("DATASET INSPECTION COMPLETED")
    print(f"Analysis outputs saved to: {OUTPUT_DIR}")

    return report


def main() -> None:
    df = load_dataset()
    analyze_dataset(df)


if __name__ == "__main__":
    main()
