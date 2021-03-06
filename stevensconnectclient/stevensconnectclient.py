# -*- coding: utf-8 -*-
"""
Stevens Connect API methods for retrieving data
Uses readings API version 3

Requires:
    python >=3.8
    requests
    pandas
    tqdm
    
License: MIT
Copyright (c) 2021 Reinert Huseby Karlsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import datetime as dt
import itertools
import re

import requests
import pandas as pd
from tqdm import tqdm


class StevensConnectSession:
    """
    Stevens Connect API methods for retrieving data

    Note:
    Data timestamps are retrieved according to the users timezone_pref_id setting
    on Stevens Connect and the setting of each particular station.
    This timezone information is stored in the index name of the returned pandas.DataFrame.
    See Stevens Connect documentation for information on timezone_pref_id.

    All data from Stevens Connect is returned, and this may include duplicated timestamps.
    See `clean_data()` for removal of these.

    Returns data from individual channels without gap markers (i.e. no nan values)

    Example usage:

    .. code-block:: python

        from stevensconnectclient import StevensConnectSession

        scs = StevensConnectSession('username@email.com', 'password')

        # get stations available, as DataFrame
        scs.get_stations()

        # information on all channels, sensors and projects available to user
        # as DataFrame
        scs.get_all_channels()

        # get sensors and channel ids under given station id
        scs.get_station_sensor_channels(station_id)

        # get data as DataFrame for given station_id, sensor_id or channel_id
        data = scs.get_data_station(station_id,
                                    start_date='2021-01-01 15:00',
                                    end_date='2021-02-01 00:00')
        data = scs.get_data_sensor(sensor_id,
                                   start_date='2021-01-01 15:00',
                                   end_date='2021-02-01 00:00')
        data = scs.get_data_channels([channel_a, channel_b],
                                     start_date='2021-01-01 15:00',
                                     end_date='2021-02-01 00:00')

        # clean data. removing duplicated timestamps,
        # check if numeric, sorting by date and time
        cleaned =  scs.clean_data(data)
    """

    API_URL_AUTH = "https://api.stevens-connect.com/authenticate"
    API_URL_PROJECT = "http://api.stevens-connect.com/project"
    API_URL_CONFIGPACKET = "https://api.stevens-connect.com/config-packet"

    def __init__(self, username, password):
        """Authenticate with username and password to Stevens Connect"""

        auth_data = {"email": username, "password": password}

        ra = requests.post(self.API_URL_AUTH, data=auth_data)
        self._auth_response = ra.json()

        if self._auth_response["errors"]:
            raise AuthenticationError(ra.status_code, self._auth_response["message"])

        auth_token = self._auth_response["data"]["token"]
        self.header_api_auth = {"Authorization": f"bearer {auth_token}"}

        # Read metadata from config packet for user
        self.cp_projects, self.cp_units = self._get_config_packet()
        self.channel_data = self._config_packet_to_df()
        self.projects = self._get_projects()

        self.timezone_pref_id = self._auth_response["data"]["user"]["timezone_pref_id"]

    def _get_projects(self):
        """Creates a dictionary with project ids and names"""
        projects = {}

        for project in self.cp_projects:
            projects[project["id"]] = project["name"]

        return projects

    def _config_packet_to_df(self):
        """Normalizes the json config-packet to a pandas DataFrame"""

        channel_norm = pd.json_normalize(
            self.cp_projects,
            record_path=["stations", "sensors", "channels"],
            meta=[
                "id",
                "name",  # project id and name
                ["stations", "id"],  # station id
                ["stations", "name"],  # station name
                ["stations", "sui"],  # station sui
                ["stations", "sensors", "name"],  # sensor name
            ],
            errors="ignore",
            meta_prefix="meta_",
        )

        # selection of data columns
        channel_data = channel_norm[
            [
                "id",
                "sensor_id",
                "name",
                "unit_id",
                "status",
                "channel_health.last_reading",
                "meta_id",
                "meta_name",
                "meta_stations.id",
                "meta_stations.name",
                "meta_stations.sui",
                "meta_stations.sensors.name",
            ]
        ]

        # rename columns with prefix
        rename_cols = {
            "id": "channel_id",
            "name": "channel_name",
            "unit_id": "channel_unit_id",
            "status": "channel_status",
            "meta_id": "project_id",
            "meta_name": "project_name",
            "meta_stations.id": "station_id",
            "meta_stations.name": "station_name",
            "meta_stations.sui": "station_sui",
            "meta_stations.sensors.name": "sensor_name",
        }

        channel_data = channel_data.rename(rename_cols, axis=1)

        # join with unit information
        cp_unit_keys = ["id", "name", "unit"]
        cp_units_df = pd.DataFrame(self.cp_units)[cp_unit_keys]
        cp_units_df.columns = [
            "channel_unit_id",
            "channel_unit_name",
            "channel_unit_str",
        ]
        channel_data = pd.merge(
            channel_data, cp_units_df, on="channel_unit_id", how="left"
        )

        channel_data = channel_data.set_index("channel_id")

        return channel_data

    def _get_config_packet(self):
        """Get the config-packet and store lists of projects and units"""

        cp_r = requests.get(self.API_URL_CONFIGPACKET, headers=self.header_api_auth)

        cp_j = cp_r.json()

        if cp_j["errors"]:
            raise APIError(cp_j.status_code, cp_j["message"])

        cp = cp_j["data"]["config_packet"]

        cp_projects = cp["projects"]
        cp_units = cp["units"]

        return cp_projects, cp_units

    def _get_station_tz(self, station_id):
        """Get timezone name of station"""

        return self.get_stations().loc[station_id, "timezone_name"]

    def get_stations(self):
        """Returns a dataframe with available stations"""

        stations = pd.json_normalize(
            self.cp_projects,
            record_path=["stations"],
            meta=["name"],
            meta_prefix="project_",
        )

        # Note that 'timezone' key appears not be used, always 0000
        stations = stations[
            [
                "id",
                "sui",
                "name",
                "project_id",
                "project_name",
                "timezone_name",
                "timezone",
            ]
        ]
        stations = stations.set_index("id")

        return stations

    def get_all_channels(self):
        """Returns a dataframe with information on all available channels"""

        return self.channel_data.copy()

    def get_station_sensor_channels(self, station_id):
        """Returns sensors and data channels available for a given station id"""

        station_id_locs = self.channel_data.station_id == station_id
        get_keys = [
            "channel_name",
            "channel_unit_id",
            "channel_unit_name",
            "channel_unit_str",
            "sensor_id",
            "sensor_name",
            "station_id",
            "station_name",
            "station_sui",
            "project_id",
            "project_name",
        ]

        station_channels = self.channel_data.loc[station_id_locs, get_keys]

        return station_channels

    def get_sensor_channels(self, sensor_id):
        """Returns data channels for a given sensor id"""

        sensor_id_locs = self.channel_data.sensor_id == sensor_id
        get_keys = [
            "channel_name",
            "channel_unit_id",
            "channel_unit_name",
            "channel_unit_str",
            "sensor_id",
            "sensor_name",
        ]

        sensor_channels = self.channel_data.loc[sensor_id_locs, get_keys]
        sensor_channels.set_index("channel_id", inplace=True)

        return sensor_channels

    def get_data_channels(
        self, channel_ids, start_datetime, end_datetime, label_names=False
    ):
        """
        Get data for a channel or list of channels within the date range
        start_datetime:end_datetime.

        Parameters
        ----------
        channel_ids : int or list
            Channel id or list of multiple channel ids.
        start_datetime : str
            Datetime string for starting time of query, format "%Y-%m-%d %H:%M".
            Timezone should depend on the user settings, which defaults to station time.
        end_datetime : str
            Datetime string for end time of query, format "%Y-%m-%d %H:%M".
            Timezone should depend on the user settings, which defaults to station time.
        label_names : boolean, optional
            Labeling of columns with channel details rather than just id.
            Labels are of format:
            <channel_id>_<station name>_<sensor name>_<channel name>_<unit>
            For example, '12345_AmazonRiver_NiceSensor_WaterDepth_mm'
            The default is False.

        Raises
        ------
        APIError
            Raises APIError if errors are returned from remote API.

        Returns
        -------
        data_chs : pandas.DataFrame
            DataFrame with result of the query.

        """

        # check if channel_ids is iterable, if not make it a list
        try:
            iter(channel_ids)
        except TypeError:
            channel_ids = [channel_ids]

        get_data = {
            "channel_ids": "",
            "range_type": "absolute",
            "start_date": start_datetime,
            "end_date": end_datetime,
            "user_timezone": "null",
            "minutes": "null",
            "page": 1,
        }

        data_chs = pd.DataFrame()
        
        # Read single channel at the time and concat to data_chs DataFrame
        for channel_id in tqdm(channel_ids, desc="Channels", unit="channel"):
            try:
                project_id = self.channel_data.loc[channel_id, "project_id"]
            except KeyError:
                tqdm.write(f"Channel {channel_id} not found")
                continue

            # make url
            API_URL_READINGS = (
                self.API_URL_PROJECT + f"/{project_id}/readings/v3/channels"
            )
            
            # change channel_id and user_time in get_data
            get_data["channel_ids"] = str(channel_id)
           
            # set page to 1
            get_data["page"] = 1
            
            # get readings
            r_readings = requests.get(
                API_URL_READINGS, params=get_data, headers=self.header_api_auth
            )
            r_readings_json = r_readings.json()

            # check for errors or empty readings
            if r_readings_json["errors"]:
                raise APIError(
                    r_readings_json.get("status_code", None), r_readings_json["message"]
                )

            if len(r_readings_json["data"]["readings"][str(channel_id)]) == 0:
                tqdm.write(f"Channel {channel_id}: no data returned from API")
                continue

            # make series/df
            data_ch = pd.DataFrame(r_readings_json["data"]["readings"][str(channel_id)])

            # check for more pages in the api response and get these pages
            while (
                r_readings_json["data"]["paging"]["last_page"]
                - r_readings_json["data"]["paging"]["current_page"]
                > 0
            ):
                get_data["page"] += 1
                r_readings = requests.get(
                    API_URL_READINGS, params=get_data, headers=self.header_api_auth
                )
                r_readings_json = r_readings.json()
                data_p = pd.DataFrame(
                    r_readings_json["data"]["readings"][str(channel_id)]
                )

                data_ch = pd.concat([data_ch, data_p])

            data_ch["timestamp"] = pd.to_datetime(
                data_ch["timestamp"], format="%Y-%m-%d %H:%M:%S"
            )

            # Index with timestamp and continue with a pd.Series object
            data_ch.set_index("timestamp", inplace=True)
            data_ch = data_ch["reading"].copy()

            tz_n = {
                1: "UTC",
                2: "LOCAL_TIME",  # TODO can we get the timezone from query param?
                3: self._get_station_tz(
                    self.channel_data.loc[channel_id, "station_id"]
                ),
            }

            # Author has noticed issues with the time conversion, print warning if
            # setting is different that station time (id 3)
            if self.timezone_pref_id in [1, 2]:
                tqdm.write('WARN: Timezone setting of user is UTC or Local time. '
                    'The API has previously had issues converting timestamps '
                    'correctly, please check the returned timestamps for correctness.')
            
            data_ch.index.name = data_ch.index.name + "_" + tz_n[self.timezone_pref_id]

            if label_names:
                label_l = self.channel_data.loc[
                    channel_id,
                    ["station_name", "sensor_name", "channel_name", "channel_unit_str"],
                ].tolist()
                
                label_l.insert(0, str(channel_id))
                reading_label = "_".join(label_l)
                
                # replace certain characters
                reading_label = (
                    reading_label.replace(" ", "_")
                    .replace("-", "_")
                    .replace(".", "_")
                    .replace("/", "_")
                    .replace("??", "2")
                    .replace("??", "3")
                )
                
                # remove all special characters (degree symbol, superscripts)
                reading_label = re.sub(r'\W+', '', reading_label)
                
                reading_label = self._remove_repeated_chars(reading_label, ["_"])
            else:
                reading_label = str(channel_id)

            data_ch.name = reading_label

            # join to all with outer
            # KeyError on join indicates different timezones for channels
            # First join is on empty frame, so set the index.name same for this

            if len(data_chs) == 0:
                data_chs.index.name = data_ch.index.name

            try:
                data_chs = data_chs.join(data_ch, on=data_ch.index.name, how="outer")
            except KeyError:
                raise ValueError(
                    f"Cannot join channels with different timestamp timezone. "
                    f"Channel id {channel_id} index name {data_ch.index.name}. "
                    f"Previous channels have {data_chs.index.name}."
                )

        # If later channels have additional data timestamps these are not joined
        # properly, and therefore get NaT in index and wrong name on index
        if data_chs.index.isna().sum() > 0:
            tqdm.write(
                f"WARN: Channels have different timestamps for {data_chs.index.isna().sum()}"
                " records causing bad index after joining channels. Attempting reindexing"
            )

            data_chs = data_chs.set_index(data_ch.index.name)
            if data_chs.index.isna().sum() > 0:
                tqdm.write(
                    f"WARN: Index still contains missing datetimes, dropping "
                    f"{data_chs.index.isna().sum()} records"
                )
                # Drop rows with NaT on index, and drop column with index name if present
                data_chs = data_chs.loc[data_chs.index.notna()]
                try:
                    data_chs = data_chs.drop(data_ch.index.name)
                except KeyError:
                    pass

        return data_chs

    def get_data_station(
        self,
        station_id,
        start_datetime,
        end_datetime,
        label_names=False,
        ignore_station_health=True,
    ):
        """
        Get data for a station within the date range start_datetime:end_datetime.
        A station can have several sensors and channels, all are returned here.

        Parameters
        ----------
        station_id : int
            Station integer id.
        start_datetime : str
            Datetime string for starting time of query, format "%Y-%m-%d %H:%M".
            Timezone should depend on the user settings, which defaults to station time.
        end_datetime : str
            Datetime string for end time of query, format "%Y-%m-%d %H:%M".
            Timezone should depend on the user settings, which defaults to station time.
        label_names : boolean, optional
            Labeling of columns with channel details rather than just id.
            Labels are of format:
            <channel_id>_<station name>_<sensor name>_<channel name>_<unit>
            For example, '12345_AmazonRiver_NiceSensor_WaterDepth_mm'
            The default is False.
        ignore_station_health : boolean, optional
            Ignore the station health data column.
            The default is True.

        Returns
        -------
        data_station : pandas.DataFrame
            DataFrame with result of the query.

        """

        channel_ids = self.channel_data.loc[
            self.channel_data.station_id == station_id
        ].index.tolist()

        if ignore_station_health:
            # only keep channels with name not containing 'station health'
            keep_cn_b = ~(
                self.channel_data.loc[channel_ids]["channel_name"].str.contains(
                    "Station Health"
                )
            )
            channel_ids = keep_cn_b.loc[keep_cn_b].index.tolist()

        if len(channel_ids) == 0:
            print(f"No channels found for station id {station_id}")
            return

        data_station = self.get_data_channels(
            channel_ids, start_datetime, end_datetime, label_names
        )

        return data_station

    def get_data_sensor(
        self, sensor_id, start_datetime, end_datetime, label_names=False
    ):
        """
        Get data for a sensor within the date range start_datetime:end_datetime.
        A sensor can have several channels, all are returned here.


        Parameters
        ----------
        sensor_id : int
            Sensor integer id.
        start_datetime : str
            Datetime string for starting time of query, format "%Y-%m-%d %H:%M".
            Timezone should depend on the user settings, which defaults to station time.
        end_datetime : str
            Datetime string for end time of query, format "%Y-%m-%d %H:%M".
            Timezone should depend on the user settings, which defaults to station time.
        label_names : boolean, optional
            Labeling of columns with channel details rather than just id.
            Labels are of format:
            <channel_id>_<station name>_<sensor name>_<channel name>_<unit>
            For example, '12345_AmazonRiver_NiceSensor_WaterDepth_mm'
            The default is False.

        Returns
        -------
        data_sensor : pandas.DataFrame
            DataFrame with result of the query.

        """

        channel_ids = self.channel_data.loc[
            self.channel_data.sensor_id == sensor_id
        ].index.tolist()

        if len(channel_ids) == 0:
            print(f"No channels found for sensor id {sensor_id}")
            return

        data_sensor = self.get_data_channels(
            channel_ids, start_datetime, end_datetime, label_names
        )

        return data_sensor

    def get_latest_datapoint_channel(self, channel_id,
                                     start_datetime=None, end_datetime=None):
        """
        Get the timestamp of the latest datapoint on a given channel_id, searching
        in window between start_datetime and end_datetime.

        Parameters
        ----------
        channel_id : int
            Channel ID in Stevens Connect.
        start_datetime : str, optional
            Datetime string for starting time of query, format "%Y-%m-%d %H:%M".
            Timezone should depend on the user settings, which defaults to station time.
            If None the start of the window is set 90 days before the time of the query.
            The default is None.
        end_datetime : str, optional
            Datetime string for end time of query, format "%Y-%m-%d %H:%M".
            Timezone should depend on the user settings, which defaults to station time.
            If None the end of the window is set to the time of the query,
            i.e. current user time.
            The default is None.

        Returns
        -------
        String datetime of the latest datapoint for channel, formatted as %Y-%m-%d %H:%M".

        """
        
        if start_datetime is None:
            start_datetime = ((dt.datetime.now() - dt.timedelta(days=90))
                              .strftime("%Y-%m-%d %H:%M"))
        if end_datetime is None:
            end_datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

        qd = self.get_data_channels(channel_id, start_datetime, end_datetime)
        
        try:
            last_dt_str = qd.index.max().strftime("%Y-%m-%d %H:%M")
            return last_dt_str
        except AttributeError:
            print('No datetime index returned from query.')
            return
            
    @staticmethod
    def clean_data(input_data):
        """
        Basic sanitation and check of data.

        The following checks and adjustments are made:
        - Check if index is all dates: Throws exception if not
        - Check if data is numeric: Converts to numeric if not
        - Drops duplicate indicies
        - Sorts data by index


        Parameters
        ----------
        input_data : pandas.Series or pandas.DataFrame
            The time series input data to be cleaned.

        Raises
        ------
        NotImplementedError
            If index is not all dates, no corrections implemented.

        Returns
        -------
        clean_data : pandas.Series or pandas.DataFrame
            Cleaned data.

        """

        clean_data = input_data.copy()

        # Drop rows with NaT on index
        if clean_data.index.isna().sum() > 0:
            clean_data = clean_data.loc[clean_data.index.notna()]

        # HINT not tested for alternative types to datetime64
        if not input_data.index.inferred_type == "datetime64":
            raise NotImplementedError(
                "Data index is not all dates. Solution not yet implemented"
            )

        # check that all columns are numeric dtype
        # for dataframes and series
        try:
            data_is_numeric = input_data.dtypes.apply(
                pd.api.types.is_numeric_dtype
            ).all()
        except AttributeError:
            data_is_numeric = pd.api.types.is_numeric_dtype(input_data.dtype)

        # convert to numeric data if dtypes are not numeric
        if not data_is_numeric:
            clean_data = pd.to_numeric(clean_data, errors="coerce")
        # drop duplicate indicies
        clean_data = StevensConnectSession._drop_duplicates(clean_data)

        # sort by index
        clean_data = clean_data.sort_index()

        return clean_data

    @staticmethod
    def _drop_duplicates(input_data):
        """Drop duplicated indicies from dataframe or series"""
        duplicated_index = input_data.index.duplicated()

        if duplicated_index.sum() > 0:
            df_nd = input_data.loc[~input_data.index.duplicated(keep="first")]
            return df_nd
        else:
            return input_data.copy()

    def _remove_repeated_chars(self, in_str, char_list):
        output = "".join(
            k if k in char_list else "".join(v) for k, v in itertools.groupby(in_str)
        )

        return output


class AuthenticationError(Exception):
    """Exception for authentication errors to Stevens Connect

    Attributes
    ----------
    status_code : int
        status code from api
    message : str
        explanation of the error
    """

    def __init__(self, status_code, message):

        message_string = message.get("errors", {}).get("email", "Unknown Error")

        self.status_code = status_code
        self.message = message_string
        super().__init__(self.message)

    def __str__(self):
        return f"{self.status_code}: {self.message}"


class APIError(AuthenticationError):
    """General exception for Stevens Connect API error

    Attributes
    ----------
    status_code : int
        status code from api
    message : str
        explanation of the error
    """

    def __init__(self, status_code, message):

        message_string = message.get("message", "Unknown error")

        self.status_code = status_code
        self.message = message_string
