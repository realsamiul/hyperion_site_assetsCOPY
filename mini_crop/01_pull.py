import sys
sys.path.append('..')
from common.satellite_pull import FixedSatelliteAcquisition
from common.config import CROP_AOI

SEARCH_PERIODS = {
    "2023":{"start":"2023-01-01","end":"2023-12-31"}
}

def main():
    FixedSatelliteAcquisition(CROP_AOI, "crop", SEARCH_PERIODS).run()

if __name__ == "__main__":
    main()