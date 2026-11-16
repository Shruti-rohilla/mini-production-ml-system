from typing import Literal, Optional

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"] = Field(alias="Gender")
    senior_citizen: YesNo = Field(alias="Senior Citizen")
    partner: YesNo = Field(alias="Partner")
    dependents: YesNo = Field(alias="Dependents")
    tenure_months: int = Field(alias="Tenure Months", ge=0, le=100)
    phone_service: YesNo = Field(alias="Phone Service")
    multiple_lines: str = Field(alias="Multiple Lines")
    internet_service: str = Field(alias="Internet Service")
    online_security: str = Field(alias="Online Security")
    online_backup: str = Field(alias="Online Backup")
    device_protection: str = Field(alias="Device Protection")
    tech_support: str = Field(alias="Tech Support")
    streaming_tv: str = Field(alias="Streaming TV")
    streaming_movies: str = Field(alias="Streaming Movies")
    contract: str = Field(alias="Contract")
    paperless_billing: YesNo = Field(alias="Paperless Billing")
    payment_method: str = Field(alias="Payment Method")
    monthly_charges: float = Field(alias="Monthly Charges", ge=0)
    total_charges: Optional[float] = Field(default=None, alias="Total Charges", ge=0)

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float
    model_used: str