import numpy as np
import pandas as pd
from typing import List, Tuple, Union

from spa.core import spa
from spa.properties import OutOfDataException, GenericProperty, GenericThreshold

_T_NUMBER = Union[int, float]


class ThresholdPropertyDF(GenericThreshold, GenericProperty):
    """Property comparison model to compare a single execution property against a threshold.
    Table 1, item 1 in the paper https://doi.org/10.1145/3613424.3623785.
    Modeled to use a pandas DataFrame as input data."""

    tag = None
    """The tag to use to extract the data from the dataframe."""

    def set_df_tag(self, tag: str):
        """Set the tag to use for the dataframe. This is used to extract the data from the dataframe.
        :param tag: The tag to use.
        """
        self.tag = tag

    def _get_tagged_data_from_df(self, data: pd.DataFrame) -> List[_T_NUMBER]:
        if self.tag is None:
            raise ValueError('Dataframe tag must be set using `set_df_tag` before calling this function')

        # Extract all cycles data from the dataframe
        filtered_df = data[data['tag'] == 'cycles']
        values = filtered_df['value'].tolist()
        return values

    def start_point_estimate(self, data: pd.DataFrame, proportion: float) -> float:
        """When using SPA to find a confidence interval, an initial point estimate is needed. This method estimates a
        starting point for the property's true value. In this case, the proportion can be thought of as the inverse of
        quantile.
        :param data: The data to use for the estimate.
        :param proportion: The proportion of the data to use for the estimate.
        :return: The estimated starting point.
        """

        values = self._get_tagged_data_from_df(data)

        # Return the quantile of the data
        return np.quantile(values, 1 - proportion)

    def extract_value(self, data: pd.DataFrame) -> Tuple[_T_NUMBER, pd.DataFrame]:
        """Extract the value from the input data. Meant to be used in conjunction with :function check_sample_satisfy:
        For this property, return only the leftmost value in the data. Returns the value and the remaining data.
        :param data: The data to extract from.
        :return: The extracted value(s).
        """

        values = self._get_tagged_data_from_df(data)

        if len(values) < 1:
            raise OutOfDataException
        # Read data from left to right

        value = values[0]

        # Drop the first value from the dataframe
        data_rows = data['tag'] == 'cycles'
        data = data.drop(data[data_rows].index[0])

        return value, data

    def check_sample_satisfy(self, value: _T_NUMBER) -> bool:
        """Check if the property is satisfied or not satisfied by the given value. Meant to be used with the
        :function extract_value: method.
        In this case, the property is satisfied if the value comparison against the threshold is True.
        :param value: The value(s) to check.
        :return: True if the property is satisfied, False otherwise.
        """
        # First ensure that the property is set
        if not (isinstance(self.threshold, int) or isinstance(self.threshold, float)):
            raise TypeError('Threshold must be an integer or float')
        # Use the comparison operator defined in the constructor to check the value against the threshold
        return self._comparison(value, self.threshold)

    def verify_data(self, data: pd.DataFrame):
        if type(data) is not pd.DataFrame:
            raise TypeError('Data must be a pandas DataFrame')


def threshold_property_with_cycles_dataframe():
    df = pd.read_csv('../tagged-data/cycles-runs.csv')

    prepared_property = ThresholdPropertyDF()
    prepared_property.set_df_tag('cycles')
    ci = spa(df, prepared_property, 0.9, 0.9)
    print(ci)


if __name__ == '__main__':
    threshold_property_with_cycles_dataframe()

