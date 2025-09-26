import pytest
from crmonitor.mpr.prediction import SamplingDimension, SamplingError, SamplingOrder
from crmonitor.mpr.prediction.sampling_x_dimensional import DimensionData, LonLatData


class TestDimensionData:
    """Test cases for DimensionData generic class."""

    def test_initialization_with_all_values(self):
        """Test initialization with all kinematic orders."""
        data = DimensionData(position=1.0, velocity=2.0, acceleration=3.0)
        assert data.position == 1.0
        assert data.velocity == 2.0
        assert data.acceleration == 3.0

    def test_initialization_with_partial_values(self):
        """Test initialization with only some kinematic orders."""
        data = DimensionData(position=1.0, velocity=2.0)
        assert data.position == 1.0
        assert data.velocity == 2.0

        with pytest.raises(SamplingError, match="acceleration.*not available"):
            _ = data.acceleration

    def test_from_dict_factory_method(self):
        """Test creation from dictionary."""
        dict_data = {
            SamplingOrder.POSITION: 1.0,
            SamplingOrder.VELOCITY: 2.0,
            SamplingOrder.ACCELERATION: 3.0,
        }
        data = DimensionData.from_dict(dict_data)

        assert data.position == 1.0
        assert data.velocity == 2.0
        assert data.acceleration == 3.0

    def test_has_order(self):
        """Test has_order method."""
        data = DimensionData(position=1.0, velocity=2.0)

        assert data.has_order(SamplingOrder.POSITION) is True
        assert data.has_order(SamplingOrder.VELOCITY) is True
        assert data.has_order(SamplingOrder.ACCELERATION) is False

    def test_get_order(self):
        """Test get_order method."""
        data = DimensionData(position=1.0, velocity=2.0)

        assert data.get_order(SamplingOrder.POSITION) == 1.0
        assert data.get_order(SamplingOrder.VELOCITY) == 2.0
        assert data.get_order(SamplingOrder.ACCELERATION) is None

    def test_missing_data_access(self):
        """Test that accessing missing data raises SamplingError."""
        data = DimensionData(position=1.0)

        with pytest.raises(SamplingError, match="velocity.*not available"):
            _ = data.velocity

        with pytest.raises(SamplingError, match="acceleration.*not available"):
            _ = data.acceleration


class TestLonLatData:
    """Test cases for LonLatData generic class."""

    def test_initialization(self):
        """Test initialization with lon and lat DimensionData."""
        lon_data = DimensionData(position=1.0, velocity=2.0)
        lat_data = DimensionData(position=3.0, velocity=4.0)

        lonlat_data = LonLatData(lon=lon_data, lat=lat_data)

        assert lonlat_data.lon == lon_data
        assert lonlat_data.lat == lat_data

    def test_from_dict_factory_method(self):
        """Test creation from nested dictionary."""
        dict_data = {
            SamplingDimension.LON: {SamplingOrder.POSITION: 1.0, SamplingOrder.VELOCITY: 2.0},
            SamplingDimension.LAT: {SamplingOrder.POSITION: 3.0, SamplingOrder.VELOCITY: 4.0},
        }

        lonlat_data = LonLatData.from_dict(dict_data)

        assert lonlat_data.lon.position == 1.0
        assert lonlat_data.lon.velocity == 2.0
        assert lonlat_data.lat.position == 3.0
        assert lonlat_data.lat.velocity == 4.0

    def test_get_dimension(self):
        """Test get_dimension method."""
        lon_data = DimensionData(position=1.0)
        lat_data = DimensionData(position=2.0)
        lonlat_data = LonLatData(lon=lon_data, lat=lat_data)

        assert lonlat_data.get_dimension(SamplingDimension.LON) == lon_data
        assert lonlat_data.get_dimension(SamplingDimension.LAT) == lat_data
