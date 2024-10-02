from pathlib import Path
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from astropy.table import Table, join, unique
import astropy.units as u
import gzip
import requests
from astroquery.vizier import Vizier
from io import TextIOWrapper
from argparse import ArgumentParser
from erfa import ErfaWarning
import logging
import warnings

log = logging.getLogger("main")

# taken from ctapipe_io_lst, converted using pyproj from official ctao coordinates
LST1_HEIGHT = 2199.885
# rounded to 6 digits
LST1_LON = -17.891497
LST1_LAT = 28.761526


warnings.simplefilter("ignore", ErfaWarning)

parser = ArgumentParser()
parser.add_argument("-t", "--time", type=Time, help="Observation time, if not given, the default is 'now'")
parser.add_argument("-m", "--max-magnitude", default=2.0, help="Maximum allowed magnitude", type=float)
parser.add_argument("-v", "--verbose", default=0, action="count")
parser.add_argument("--min-zenith", default=5, type=float, help="Minimum zenith distance in degrees")
parser.add_argument("--max-zenith", default=40, type=float, help="Minimum zenith distance in degrees")
parser.add_argument("--observatory-name", help="Name of observatory, if not given, will use lat/lon/height options")
parser.add_argument("--lon", default=LST1_LON, help="Observatory longitude in degree, default is LST-1 location")
parser.add_argument("--lat", default=LST1_LAT, help="Observatory latitude in degree, default is LST-1 location")
parser.add_argument("--height", default=LST1_HEIGHT, help="Observatory heiht in meter, default is LST-1 location")


CACHE_FILE = Path("~/.psf_stars.ecsv").expanduser()


def read_ident_name_file(ident=6, colname="name"):
    log.info(f"Downloading common identifcation file {ident}")
    r = requests.get(f"https://cdsarc.cds.unistra.fr/ftp/I/239/version_cd/tables/ident{ident}.doc.gz", stream=True)
    r.raise_for_status()

    with r, gzip.GzipFile(fileobj=r.raw, mode="r") as gz:
        table = {"HIP": [], colname: []}

        for line in TextIOWrapper(gz):
            name, hip = line.split("|")
            table["HIP"].append(int(hip.strip()))
            table[colname].append(name.strip())

    return Table(table)


def get_hipparcos_stars(max_magnitude):
    stars = None

    if CACHE_FILE.exists():
        log.info("Loading stars from cached table")
        try:
            stars = Table.read(CACHE_FILE)
            if stars.meta["max_magnitude"] >= max_magnitude:
                log.debug(f"Loaded table is valid for {max_magnitude = }")
            else:
                log.debug("Loaded cache table has smaller max_magnitude, reloading")
                stars = None
        except Exception:
            log.exception("Cache file exists but reading failed. Recreating")

    if stars is None:
        log.info("Querying Vizier for Hipparcos catalog")
        # query vizier for stars with 0 <= Vmag <= max_magnitude
        hipparcos_catalog = "I/239/hip_main"
        vizier = Vizier(
            catalog=hipparcos_catalog,
            columns=["HIP", "RAICRS", "DEICRS", "pmRA", "pmDE", "Vmag"],
            row_limit=1000000,
        )
        stars = vizier.query_constraints(Vmag=f"0.0..{max_magnitude}")[0]

        # add the nice names
        common_names = read_ident_name_file(ident=6)
        flamsteed_designation = read_ident_name_file(ident=4, colname="flamsteed")

        common_names = join(common_names, flamsteed_designation, keys="HIP", join_type="outer")

        # multiple flamsteed per source, only use one
        common_names = unique(common_names, keys="HIP")


        stars = join(stars, common_names, keys="HIP", join_type="left")

        stars.meta["max_magnitude"] = max_magnitude
        stars.write(CACHE_FILE, overwrite=True)

    stars = stars[stars["Vmag"] < max_magnitude]
    # add a column with a skycoord object
    stars["icrs"] = SkyCoord(
        ra=stars["RAICRS"].quantity,
        dec=stars["DEICRS"].quantity,
        pm_ra_cosdec=stars["pmRA"].quantity,  # yes, pmRA is already pm_ra_cosdec
        pm_dec=stars["pmDE"].quantity,
        frame="icrs",
        obstime=Time("J1991.25"),
    )


    return stars


@u.quantity_input(min_zenith=u.deg, max_zenith=u.deg)
def get_psf_stars(stars, obstime, location, min_zenith=5 * u.deg, max_zenith=45 * u.deg):
    # get current ICRS coordinates
    icrs_now = stars["icrs"].apply_space_motion(obstime)

    # astropy throws an error on conversion from icrs to altaz if proper motion is there
    icrs_now = SkyCoord(ra=icrs_now.ra, dec=icrs_now.dec, obstime=obstime)

    # transform to AltAz
    frame = AltAz(obstime=obstime, location=location)
    altaz = icrs_now.transform_to(frame)

    # find low zenith, bright stars
    valid_zd = (altaz.zen > min_zenith) & (altaz.zen < max_zenith)
    candidates = stars[valid_zd].copy()
    candidates["alt"] = altaz.alt[valid_zd].to(u.deg)
    candidates["az"] = altaz.az[valid_zd].to(u.deg)
    candidates["zd"] = altaz.zen[valid_zd].to(u.deg)

    candidates.sort(["Vmag", "zd"])
    for col in ('alt', 'az', 'zd'):
        candidates[col].info.format = ".1f"
    return candidates[["HIP", "name", "flamsteed", "Vmag", "alt", "az", "zd"]]


def main(args=None):
    args = parser.parse_args(args)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose == 2:
        level = logging.DEBUG

    logging.basicConfig(level=level, datefmt="%Y-%m-%dT%H:%M:%S", format="%(asctime)s.%(msecs)d|%(levelname)s|%(message)s")

    stars = get_hipparcos_stars(args.max_magnitude)

    if args.observatory_name is not None:
        location = EarthLocation.of_site(args.observatory_name)
    else:
        location = EarthLocation(lon=args.lon * u.deg, lat=args.lat * u.deg, height=args.height * u.m)

    if args.time is None:
        args.time = Time.now()

    log.info(f"Potential PSF stars for {args.time.isot}")
    psf_stars = get_psf_stars(
        stars=stars,
        obstime=args.time,
        location=location,
        min_zenith=args.min_zenith * u.deg,
        max_zenith=args.max_zenith * u.deg,
    )
    print(psf_stars)


if __name__ == "__main__":
    main()
