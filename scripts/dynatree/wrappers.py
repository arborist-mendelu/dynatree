from dynatree.dynatree import DynatreeMeasurement
from pydantic import BaseModel, Field, validator
import os


class DynatreeMeasurementWrapper(BaseModel):
    """
    Wrapper pro `DynatreeMeasurement` s Pydantic validací a výchozími hodnotami.

    >>>import dynatree.wrappers as wr
    >>>m = wr.DynatreeMeasurementWrapper(tree=8).to_dynatree()
    >>>m.data_pulling
    """

    day: str = Field(default="2021-03-22", description="Den měření")
    tree: str | int = Field(default="BK01", description="Číslo stromu")
    measurement: str | int = Field(default="M01", description="Označení měření")
    measurement_type: str = Field(default="normal", description="Typ měření")
    datapath: str = Field(default_factory=lambda: os.getenv("DYNATREE_DATAPATH", "../data"))

    @validator("day", pre=True)
    def normalize_day(cls, value):
        return value.replace("_", "-")

    @validator("tree", pre=True)
    def normalize_tree(cls, value):
        if isinstance(value, int) or str(value).isdigit():
            return f"{int(value):02}"
        return str(value)

    @validator("measurement", pre=True)
    def normalize_measurement(cls, value):
        return f"{value}" if isinstance(value, int) else str(value)

    @validator("measurement_type", pre=True)
    def normalize_measurement_type(cls, value):
        return value.lower()

    def to_dynatree(self) -> "DynatreeMeasurement":
        """Vrátí instanci původní třídy `DynatreeMeasurement` s validovanými a výchozími daty."""
        return DynatreeMeasurement(
            day=self.day,
            tree=self.tree,
            measurement=self.measurement,
            measurement_type=self.measurement_type,
            datapath=self.datapath
        )
