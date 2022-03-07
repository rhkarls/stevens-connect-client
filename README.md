# stevens-connect-client
Python client for Stevens Connect cloud data acquisition system

API methods for retrieving station metadata and data to pandas DataFrames

Requires:
    `python >= 3.8`
    `requests`
    `pandas`
    `tqdm`

## Example usage


```python
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
```

## Methods:

See doc strings for methods for more details

`get_stations`: Returns a dataframe with available stations.

`get_all_channels`: Returns a dataframe with information on all available channels.

`get_station_sensor_channels`: Returns sensors and data channels available for a given station id.

`get_sensor_channels`: Returns data channels for a given sensor id.

`get_data_channels`: Get data for a channel or list of channels.

`get_data_station`: Get data for a station. A station can have several sensors and channels, all are returned here.

`get_data_sensor`: Get data for a sensor. A sensor can have several channels, all are returned here.

`get_latest_datapoint_channel`: Get the timestamp of the latest datapoint on a given channel.

`clean_data`: Static method. Basic sanitation and check of data.