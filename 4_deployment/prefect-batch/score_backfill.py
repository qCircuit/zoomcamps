from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import flow
from score_original import ride_duration_prediction

@flow
def ride_duration_prediction_backfill():
    start_date = datetime(year=2021, month=3, day=1)
    end_date = datetime(year=2021, month=4, day=1)

    d = start_date

    while d < end_date:
        ride_duration_prediction(
            TAXI_TYPE="green",
            RUN_ID="ac7d9665277c494da7929f136c28faf9",
            run_date = d
        )

        d = d + relativedelta(months=1)

if __name__ == "__main__":
    ride_duration_prediction_backfill()