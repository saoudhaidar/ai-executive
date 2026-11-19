class BackendUnavailable(Exception):
    pass


SYNTHETIC_CUSTOMERS = {
    "C-1001": {"resident": True, "annual_income": 32_000},
    "C-1002": {"resident": True, "annual_income": 78_000},
    "C-1003": {"resident": False, "annual_income": 24_000},
}
SYNTHETIC_CUSTOMER_IDS = frozenset(SYNTHETIC_CUSTOMERS)


class MockEligibilityAPI:
    name = "mock_eligibility_api"

    def __init__(self):
        self.customers = {key: value.copy() for key, value in SYNTHETIC_CUSTOMERS.items()}
        self.call_count = 0

    def verify(self, customer_id, service_code):
        self.call_count += 1
        customer = self.customers.get(customer_id)
        if customer is None:
            return {
                "eligible": None,
                "outcome": "NOT_FOUND",
                "reason": "No matching synthetic customer record was found.",
                "decision_source": self.name,
            }
        if service_code != "RENTAL_ASSISTANCE":
            return {
                "eligible": False,
                "reason": "This mock eligibility service only supports rental assistance.",
                "decision_source": self.name,
            }
        if (
            not isinstance(customer, dict)
            or type(customer.get("resident")) is not bool
            or type(customer.get("annual_income")) is not int
            or customer["annual_income"] < 0
        ):
            return {
                "eligible": None,
                "outcome": "UNVERIFIABLE",
                "reason": "The synthetic customer record is incomplete or malformed.",
                "decision_source": self.name,
            }
        eligible = customer["resident"] and customer["annual_income"] <= 50_000
        reason = (
            "The mock record meets the residency and income checks."
            if eligible
            else "The mock record does not meet the residency and income checks."
        )
        return {
            "eligible": eligible,
            "outcome": "ELIGIBLE" if eligible else "INELIGIBLE",
            "reason": reason,
            "decision_source": self.name,
        }


class MockServiceRequestAPI:
    name = "mock_service_request_api"

    def __init__(self):
        self.call_count = 0
        self._results = {}
        self.fail_next = False

    def create(self, idempotency_key, customer_id, service_code):
        if idempotency_key in self._results:
            return self._results[idempotency_key]
        if self.fail_next:
            self.fail_next = False
            raise BackendUnavailable("The mock request service is unavailable.")
        self.call_count += 1
        result = {
            "external_reference": f"SR-{self.call_count:04d}",
            "external_status": "RECEIVED",
        }
        self._results[idempotency_key] = result
        return result
