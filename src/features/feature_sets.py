TARGET_REGRESSION = "ADR"
TARGET_CLASSIFICATION = "IsCanceled"

BOOKING_TIME_FEATURES = [
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
    "DepositType",
    "Agent",
    "Company",
    "CustomerType",
]

EXCLUDED_FEATURES = [
    "ReservationStatus",
    "ReservationStatusDate",
    "AssignedRoomType",
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]

ADR_MAX = 1000

CLUSTERING_NUMERIC_FEATURES = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "IsRepeatedGuest",
]

CLUSTERING_CATEGORICAL_FEATURES = [
    "MarketSegment",
    "DepositType",
    "CustomerType",
    "DistributionChannel",
]

CLUSTERING_FEATURES = CLUSTERING_NUMERIC_FEATURES + CLUSTERING_CATEGORICAL_FEATURES
