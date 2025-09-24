import sys
sys.path.append('..')
from common.satellite_pull import FixedSatelliteAcquisition
from common.config import EROSION_AOI

SEARCH_PERIODS = {
    "1990":{"start":"1990-01-01","end":"1990-12-31"},
    "2023":{"start":"2023-01-01","end":"2023-12-31"}
}

def main():
    FixedSatelliteAcquisition(EROSION_AOI, "erosion", SEARCH_PERIODS).run()

if __name__ == "__main__":
    main()