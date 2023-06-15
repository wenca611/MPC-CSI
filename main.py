"""
Weather analyzer

Name: main.py
Description: Statistical weather analyser for the Czech Republic.
Autor: Václav Pastušek
Creation date: 11. 2. 2023
Last update: 29. 4. 2023
School: BUT FEEC
VUT number: 204437
Python version: 3.9.13
"""

# Standard libraries:
try:
    import platform  # Access to underlying platform’s identifying data
    import os  # Miscellaneous operating system interfaces
    from typing import Union, Callable, Optional, Generator  # Support for type hints
    import functools  # Higher-order functions and operations on callable objects
    import itertools  # Functions creating iterators for efficient looping
    import signal  # Set handlers for asynchronous events
    import re  # Regular expression operations
    import time as tim  # Time access and conversions
    import concurrent.futures  # Launching parallel tasks
except ImportError as L_err:
    print("Chyba v načtení standardní knihovny: {0}".format(L_err))
    exit(1)
except Exception as e:
    print(f"Jiná chyba v načtení standardní knihovny: {e}")
    exit(1)

# Third-party libraries:
try:
    # Libraries for web scraping
    import requests  # HTTP library for sending requests
    from bs4 import BeautifulSoup  # Library for parsing HTML and XML documents

    # Data manipulation and analysis
    import xarray as xr  # Library for working with labeled multi-dimensional arrays
    import numpy as np  # Fundamental package for scientific computing
    from scipy.stats import skew, kurtosis, hmean, gmean, linregress  # Library for statistics and regression analysis
    from scipy.fftpack import dct  # Library for discrete cosine transform

    # Visualization
    import tqdm  # displaying progress bars
    import matplotlib.pyplot as plt  # Library for creating static, animated, and interactive visualizations
    import matplotlib as mpl  # Library for customization of matplotlib plots
    import matplotlib.colors as mcolors  # Library for color mapping and normalization

    # User interaction
    import keyboard  # Library for detecting keyboard input
except ImportError as L_err:
    print("Chyba v načtení knihovny třetích stran: {0}".format(L_err))
    exit(1)
except Exception as e:
    print(f"Jiná chyba v ačtení knihovny třetích stran: {e}")
    exit(1)

DEBUG_PRINT = False  # If True, print more information about the operations performed.
DEBUG_SKIP = True  # If True, skip parts for loading data into the loop to draw graphs.


class JumpException(Exception):
    """
    Custom exception class used for jumping back from a nested function to the main loop
    when there is a problem with loading a webpage.
    """
    pass


class Utils:
    """
    A utility class providing various helper functions.
    """

    @staticmethod
    def welcome() -> None:
        """
        Prints Welcome text

        :return: None
        """
        print("""============Vítejte v aplikaci pro Statistické zpracování dat počasí v ČR================
-autor: Bc. Václav Pastušek
-škola: VUT FEKT
-minimální požadovaná verze Pythonu: 3.9
-aktuální verze Pythonu: {}
-VUT číslo: 204437
-zdroje dat: https://www.chmi.cz/historicka-data/pocasi/uzemni-teploty,
             https://www.chmi.cz/historicka-data/pocasi/uzemni-srazky
-poznámky: u každého uživatelského vstupu se dá zastavit program napsáním a potvrzením 'x'
    nebo vypsat nápověda pomocí 'h'\n""".format(platform.python_version()))

    @staticmethod
    def pyhelp() -> None:
        """
        Prints Help text

        :return: None
        """
        print("""Nápověda:
pro potvrzení napište 'a' nebo 'y'
pro zamítnutí napište 'n'
pro výpis nápovědy 'h'
pro výběr plotu '0-9'
pro výběr let, př.: '2000', '1990-2000', '1982, 1984, 1986-1988'
pro výběr krajů, př.: '0', '1,2-7,12', '7-8'
a pro ukončení programu 'k', 'e', 'x' nebo '.'\n""")

    @staticmethod
    def debug(func: Callable[..., any]) -> Callable[..., any]:
        """
        A decorator function for debugging purposes.

        :param func: A function to be decorated.
        :return: The decorated function.
        """

        def wrapper(*args: tuple[any, ...], **kwargs: any) -> any:
            print("Funkce:", func.__name__) if DEBUG_PRINT else None
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def handle_keyboard_interrupt(frame: Optional[object] = None, signal: Optional[object] = None) -> None:
        """
        A function for handling keyboard interrupt signal.

        :param frame: Not used.
        :param signal: Not used.
        :return: None
        """
        _, _ = frame, signal  # to reduce warnings
        print("\nUkončeno stiskem klávesy Ctrl+C nebo tlačítkem Stop")
        exit()


class UserInterface:
    """
    The UserInterface class represents a collection of static methods used for handling user input and
    parsing it into the desired format.
    """

    @staticmethod
    def parse_date_range(date_str: str) -> list[int]:
        """
        Parses a date range in string format and returns a list of years.

        :param date_str: string with the date range
        :return: list of years in the range
        """
        years = []
        if re.match(r'(\d{4}(-\d{4})?)(, *(\d{4}(-\d{4})?))*$', date_str):
            # If the date string matches the format 'YYYY' or 'YYYY-YYYY' or 'YYYY,YYYY-YYY' etc.
            date_ranges = re.split(', |,', date_str)
            for date_range in date_ranges:
                if '-' in date_range:
                    start, end = date_range.split('-')
                    years += list(range(int(start), int(end) + 1))
                else:
                    years.append(int(date_range))

            if len(set(years)) != len(years):
                print("Roky se opakují!")
                return []

            if years != sorted(years):
                print("Roky nejsou vzestupně!")
                return []

            return years  # correct
        else:
            print("Špatný datový formát. (př.:'2000', '1990-2000', '1982, 1984, 1986-1988')")
            return []

    @staticmethod
    def parse_region_range(region_str) -> list[int]:
        """
        Parses a region range in string format and returns a list of region numbers.

        :param region_str: string with the region range
        :return: list of region numbers in the range
        """
        regions = []
        if re.match(r'(\d{1,2}(-\d{1,2})?)(, *(\d{1,2}(-\d{1,2})?))*$', region_str):
            region_ranges = re.split(', |,', region_str)
            for region_range in region_ranges:
                if '-' in region_range:
                    start, end = region_range.split('-')
                    regions += list(range(int(start), int(end) + 1))
                else:
                    regions.append(int(region_range))

            if len(set(regions)) != len(regions):
                print("Regiony se opakují!")
                return []

            return regions  # correct
        else:
            print("Špatný regionální formát. (př.:'5', '0-3', '0, 1, 10-13')")
            return []

    @staticmethod
    def input_loop(text: str, match: bool = False, year: bool = False, region: bool = False) -> Union[bool, str, list]:
        """
        Loops until valid user input is entered.

        :param text: message to display to the user
        :param match: whether the user input should be a single digit
        :param year: whether the user input should be a date range
        :param region: whether the user input should be a region range
        :return: valid user input
        """
        yes_list = ["a", "ano", "y", "yes", "souhlas", "samozřejmě", "certainly"]
        no_list = ["n", "ne", "no", "nope", "nesouhlas"]
        help_list = ["h", "help", "t", "tut", "tutorial"]
        end_list = ["k", "konec", "e", "end", "x", "."]
        while True:
            if region:
                # Prompt user for region range input
                user_input: str = input(text + " (0-13): ")
                low_user_input: str = user_input.lower()
                if low_user_input in help_list:
                    Utils.pyhelp()
                    continue
                elif low_user_input in end_list:
                    exit()
                else:
                    return UserInterface.parse_region_range(low_user_input)

            if year:
                # Prompt user for date range input
                user_input: str = input(text + " (1961-2022): ")
                low_user_input: str = user_input.lower()
                if low_user_input in help_list:
                    # Display help message
                    Utils.pyhelp()
                    continue
                elif low_user_input in end_list:
                    # Exit program if user input is in end list
                    exit()
                else:
                    # Parse and return region range input
                    return UserInterface.parse_date_range(low_user_input)

            if match:
                # Prompt user for single digit input
                user_input: str = input(text + " (0-9/help/konec)?: ")
                low_user_input: str = user_input.lower()
                if low_user_input.isdigit():
                    # Return user input if it is a single digit
                    return low_user_input
                elif low_user_input in help_list:
                    # Display help message
                    Utils.pyhelp()
                    continue
                elif low_user_input in end_list:
                    # Exit program if user input is in end list
                    exit()
            else:
                # Prompt user for yes/no input
                user_input: str = input(text + " (ano/ne/help/konec)?: ")
                low_user_input: str = user_input.lower()
                if low_user_input in yes_list:
                    # Return True if user input is in yes list
                    return True
                elif low_user_input in no_list:
                    # Return False if user input is in no list
                    return False
                elif low_user_input in help_list:
                    # Display help message
                    Utils.pyhelp()
                    continue
                elif low_user_input in end_list:
                    # Exit program if user input is in end list
                    exit()


class DataFetcher:
    """
    The DataFetcher class represents an object that is responsible for fetching weather data from the internet.
    It is capable of creating and loading data from a backup in case the internet connection is not available.
    """
    # the base URL of the CHMI website for weather data
    HIDDEN_URL: str = "https://www.chmi.cz/files/portal/docs/meteo/ok"
    # the full URL for the CHMI page containing temperature data
    TEMPER_MAIN_URL: str = HIDDEN_URL + "/uzemni_teploty_cs.html"
    # the full URL for the CHMI page containing precipitation data
    PRECIP_MAIN_URL: str = HIDDEN_URL + "/uzemni_srazky_cs.html"
    BACKUP_PATH: str = "backup"  # the backup directory path
    TIMEOUT: int = 60  # maximum timeout for fetching data from the internet in seconds

    def __init__(self) -> None:
        # Initialize class variables
        self.data: dict[any] = {}
        self.futures: set = set()
        self.temper_data_are: bool = False
        self.precip_data_are: bool = False
        self.parallel: bool = False

    def online_control(self, url: str) -> bool:
        """
        The online_control method checks the connection to the given url.

        :param url: A string representing the url to check connection.
        :return: A boolean indicating the connection status, True if the connection was successful, False otherwise.
        """
        err_flag: bool = False
        try:
            pass
            r = requests.get(url, timeout=self.TIMEOUT)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("Http Error:", errh)
            err_flag: bool = True
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:", errc)
            err_flag: bool = True
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:", errt)
            err_flag: bool = True
        except requests.exceptions.RequestException as err:
            print("Something Else", err)
            err_flag: bool = True
        return not err_flag  # T=not any error, F=any error

    def fetch_page(self, url: str, idx: int = 0, encode: str = "") -> tuple[int, str]:
        """
        Fetches the HTML content of the specified URL and returns it as a tuple along with the specified index.

        :param url: the URL to fetch the HTML content from
        :param idx: the index of the fetched content
        :param encode: the encoding type for the fetched content
        :return: a tuple containing the index and the HTML content of the fetched page
        """
        # Check if the website is online before attempting to fetch data
        if not self.online_control(self.TEMPER_MAIN_URL):
            print("Problém s webovou stránkou při načítání dat s vícero webových stránek (počkejte " +
                  str(self.TIMEOUT) + ")")
            for future in self.futures:
                future.cancel()  # Cancel running threads
            raise JumpException()

        # Fetch the HTML content from the specified URL
        response: requests.Response = requests.get(url, timeout=self.TIMEOUT)

        # Set the specified encoding type if provided
        if encode:
            response.encoding = encode

        # Return a tuple containing the index and the HTML content of the fetched page
        return idx, response.text

    def get_tables(self, urls: list[str]) -> list:
        """
        Fetches and parses HTML tables from a list of URLs.

        :param urls: a list of URLs to fetch and parse tables from
        :return: a list of parsed HTML tables
        """
        tim.sleep(0.1)  # small sleep, because tqdm is sometimes too fast

        # Create a list of None values with the length of the given list of URLs
        table: list[BeautifulSoup.element.Tag] = [None] * len(urls)

        # If parallel processing is enabled, use ThreadPoolExecutor to fetch the pages concurrently
        if self.parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit a fetch_page task for each URL and store the returned future in a set
                self.futures: set[any] = {executor.submit(self.fetch_page, url, idx, "windows-1250") for idx, url in
                                          enumerate(urls)}
                try:
                    # Use tqdm to show a progress bar while the futures are completed
                    for f in tqdm.tqdm(concurrent.futures.as_completed(self.futures), total=len(self.futures)):
                        idx: int
                        data: str
                        idx, data = f.result()
                        # Use BeautifulSoup to parse the fetched data and find the table element
                        soup: BeautifulSoup = BeautifulSoup(data, "html.parser")
                        table[idx]: BeautifulSoup.element.Tag = soup.body.find("table")
                except Exception as e:
                    # If an exception occurs, cancel all running futures and raise a JumpException
                    print(f"An exception occurred: {e}")
                    for f in self.futures:
                        f.cancel()
                    raise JumpException()

        # If parallel processing is disabled, fetch the pages sequentially and use tqdm to show a progress bar
        else:
            for idx, url in enumerate(tqdm.tqdm(urls, total=len(urls))):
                idx: int
                data: str
                idx, data = self.fetch_page(url, idx, "windows-1250")
                soup: BeautifulSoup = BeautifulSoup(data, "html.parser")
                table[idx]: BeautifulSoup.element.Tag = soup.body.find("table")

        # Return the list of fetched tables
        return table

    def get_weather_data(self, temper_or_not_precip: bool) -> xr.DataArray:
        """
        Fetches average monthly air temperature or monthly precipitation data in comparison with normal values for
        the territory of the Czech Republic and its regions.

        :param temper_or_not_precip: A boolean flag that specifies whether to fetch temperature data (True) or
        precipitation data (False).
        :return: A xarray.DataArray object containing the fetched data.
        """
        temper_normal: list[str] = ["1991-2020", "1981-2010", "1961-1990"]
        temper_text: str = "Načtena Průměrná měsíční teplota vzduchu ve srovnání s normálem {} " \
                           "na území ČR a jednotlivých krajů, pro roky:"
        precip_normal: list[str] = ["1991-2020", "1981-2010", "1961-1990"]
        precip_text: str = "Načteny Měsíční úhrny srážek ve srovnání s normálem {} " \
                           "na území ČR a jednotlivých krajů, pro roky:"

        # Set the appropriate URL and text strings based on whether temperature or precipitation data is being fetched
        if temper_or_not_precip:
            weather_main_url: str = self.TEMPER_MAIN_URL
            weather_normal: list[str] = temper_normal
            weather_text: str = temper_text
        else:
            weather_main_url: str = self.PRECIP_MAIN_URL
            weather_normal: list[str] = precip_normal
            weather_text: str = precip_text

        # Fetch the HTML from the weather data main URL and parse it using BeautifulSoup
        weather_html: str
        _, weather_html = self.fetch_page(weather_main_url)
        soup: BeautifulSoup = BeautifulSoup(weather_html, "html.parser")
        body: BeautifulSoup.element.Tag = soup.body
        tables: BeautifulSoup.element.ResultSet = body.find_all("table")
        weather_part_urls: list[str] = []

        # Fetch the weather data from the various part URLs for each time period
        for i, table in enumerate(tables):
            print(weather_text.format(weather_normal[i]), end=" ")
            link_a: BeautifulSoup.element.ResultSet = table.find_all("a")
            dates: list[int] = list(map(lambda x: int(x.text), link_a))
            min_dates: int
            max_dates: int
            min_dates, max_dates = min(dates), max(dates)
            print(f"{min_dates}" if min_dates == max_dates else f"{min_dates}-{max_dates}")
            weather_part_urls.extend([a.get("href")[1:] for a in link_a])

        # Creates list of weather URLs for each region
        weather_urls: list[str] = [self.HIDDEN_URL + url_end for url_end in weather_part_urls]
        # Gets weather table data for each region
        weather_table: list[BeautifulSoup.element.Tag] = self.get_tables(weather_urls)  # size=122 in 2023

        # Converts weather table data to 3D xarray data
        xr2d_data: list[xr.DataArray] = []
        for i in range(len(weather_table)):
            rows: BeautifulSoup.element.ResultSet = weather_table[i].find_all("tr")  # all rows
            th: BeautifulSoup.element.ResultSet = rows[0].find_all("td")  # header
            header: list[str] = [th[0].text] + [""] * 14 + [th[-1].text]
            header[2:2 + 12 + 1] = [td.text for td in rows[1].find_all("td")]
            data2d: list[list[str]] = [header]
            for j, row in enumerate(rows[2:]):
                data1d = [td.text for td in row.find_all("td")]
                if j % 3 != 0:
                    data1d: list[str] = [rows[j // 3 * 3 + 2].td.text] + data1d
                data2d.append(data1d)
            xr2d_data.append(xr.DataArray(data2d, dims=("row", "col")))

        # Concatenates xarray data for each region into one 3D xarray dataset
        return xr.concat(xr2d_data, dim="time")  # time, row, col [122, 42, 15]

    def get_url_data(self) -> dict:
        """
        Fetches the weather data based on the parameters specified in the class.

        :return: A dictionary containing the weather data.
        """
        # If temperature data is to be fetched, it is added to the data dictionary.
        # If precipitation data is to be fetched, it is added to the data dictionary.
        data: dict = ({'temper': self.get_weather_data(True)} if self.temper_data_are else {}) | \
                     ({'precip': self.get_weather_data(False)} if self.precip_data_are else {})
        return data

    def destroy_backup(self) -> None:
        """
        Deletes the backup files from the backup directory for temperature and precipitation data if they exist.

        :return: None
        """
        try:
            # loop through all files in the backup directory
            for filename in os.listdir(self.BACKUP_PATH):
                if self.temper_data_are:
                    # if temperature data is being used and filename matches the temperature data file name
                    if (self.temper_data_are and filename == "data_temper.nc") or (self.precip_data_are and filename ==
                                                                                   "data_precip.nc"):
                        file_path = os.path.join(self.BACKUP_PATH, filename)
                        # remove the file
                        os.remove(file_path)
                        print(f"Soubor {file_path} byl odstraněn.")
        except Exception as e:
            # handle any exceptions that occur during file deletion
            print(f"Vyskytla se chyba při mazání souboru: {e}")

    def create_new_backup(self, data: dict) -> None:
        """
        Creates a new backup of data in NetCDF format.

        :param data: Dictionary with weather data to be backed up.
        :return: None
        """
        if "temper" in data:
            data["temper"].to_netcdf(f"{self.BACKUP_PATH}/temper.nc")
        if "precip" in data:
            data["precip"].to_netcdf(f"{self.BACKUP_PATH}/precip.nc")

    def load_backup(self) -> dict:
        """
        Loads data from the backup files and returns a dictionary containing the data.

        :return: A dictionary containing loaded temperature and precipitation data if found.
        """
        data: dict = {}
        if self.temper_data_are:
            try:
                data["temper"] = xr.open_dataset(f"{self.BACKUP_PATH}/temper.nc").squeeze().to_array().squeeze()
            except FileNotFoundError:
                print("Záloha teplot nebyla nalezena.")
        if self.precip_data_are:
            try:
                data["precip"] = xr.open_dataset(f"{self.BACKUP_PATH}/precip.nc").squeeze().to_array().squeeze()
            except FileNotFoundError:
                print("Záloha srážek nebyla nalezena.")
        return data

    def get_data(self, online: bool, temper: bool, precip: bool, parallel: bool = False) -> dict:
        """
        Fetches temperature and precipitation data from online/offline sources and returns it as a dictionary.

        :param online: Flag whether to fetch data online or not.
        :param temper: Flag whether to fetch temperature data or not.
        :param precip: Flag whether to fetch precipitation data or not.
        :param parallel: Flag whether to fetch data in parallel or not.
        :return: A dictionary containing fetched data, where keys are "temper" and/or "precip".
        """
        backup_path_create: bool = False
        backup: bool = False
        backup_path: bool = os.path.exists(self.BACKUP_PATH)
        change_backup: bool = False
        self.temper_data_are: bool = temper
        self.precip_data_are: bool = precip
        self.parallel: bool = parallel

        if not backup_path:  # html folder control
            print(f'Složka "{self.BACKUP_PATH}" pro html soubory neexistuje!')
            if not online:
                print("Chybí záloha html stránek pro offline zpracování dat")
                return {}
            backup_path_create: bool = UserInterface.input_loop("Přejete si ji vytvořit?")
            if backup_path_create:  # online
                os.makedirs(self.BACKUP_PATH)

        if backup_path or backup_path_create:  # path exists
            change_backup: bool = False
            if not backup_path_create:
                if any(f.endswith('.nc') for f in os.listdir(self.BACKUP_PATH)):  # *.html control
                    backup: bool = True
                    if online:
                        online: bool = UserInterface.input_loop("Byla nalezena offline data, " +
                                                                "přejete si pokračovat dále v online režimu?")
                    if online:
                        change_backup: bool = UserInterface.input_loop("Chcete si přepsat zálohované data?")
                else:  # no *.html, backup = False
                    if not online:  # offline
                        print("Chybí záloha html stránek pro offline zpracování dat")
                        return {}
                    else:  # online
                        change_backup: bool = UserInterface.input_loop(
                            "Nebyla nalezena záloha, přejete si ji vytvořit?")
        if online:
            # online control
            if not self.online_control(self.TEMPER_MAIN_URL):
                print("Problém s webovou stránkou")
                return {}
            if not self.online_control(self.PRECIP_MAIN_URL):
                print("Problém s webovou stránkou")
                return {}

            if backup_path and change_backup:  # get_url_data + create_new_backup
                print("get_url_data + create_new_backup") if DEBUG_PRINT else None
                data: dict = self.get_url_data()
                self.create_new_backup(data)
            elif backup:  # backup exists
                if change_backup:  # get_url_data + destroy backup + create new backup
                    print("get_url_data + destroy_backup + create_new_backup") if DEBUG_PRINT else None
                    data: dict = self.get_url_data()
                    self.destroy_backup()
                    self.create_new_backup(data)
                else:  # get_url_data + (skip backup)
                    print("get_url_data + (skip backup)") if DEBUG_PRINT else None
                    data: dict = self.get_url_data()
            else:  # get_url_data + (no backup)
                print("get_url_data") if DEBUG_PRINT else None
                data: dict = self.get_url_data()
        else:  # offline, load backup
            print("load_backup") if DEBUG_PRINT else None
            data: dict = self.load_backup()
        return data


class GraphPlotter:
    """
    Class GraphPlotter is a utility class that provides methods for plotting various types of graphs
    related to weather data.
    It can plot 2D and 3D graphs, histograms, boxplots, and correlation plots.
    It takes input data in the form of a dictionary containing temperature and/or precipitation data.
    The class also provides an interactive user interface to choose which type of graph to plot and on which data.
    """

    def __init__(self, data: dict[str, xr.DataArray], temper_choose: bool, precip_choose: bool) -> None:
        # Set the months and regions for later use
        self.months: list[str] = ['Leden', 'Únor', 'Březen', 'Duben', 'Květen', 'Červen', 'Červenec', 'Srpen', 'Září',
                                  'Říjen', 'Listopad', 'Prosinec']
        self.regions: list[str] = ['Česká republika', 'Praha a Středočeský', 'Jihočeský', 'Plzeňský', 'Karlovarský',
                                   'Ústecký', 'Liberecký', 'Královéhradecký', 'Pardubický', 'Vysočina', 'Jihomoravský',
                                   'Olomoucký', 'Zlínský', 'Moravskoslezský']
        # Save the user's choices for temperature and precipitation data
        self.temper_choose: bool = temper_choose
        self.precip_choose: bool = precip_choose
        self.last_time: float = tim.time()
        # Make a copy of the original data to use as a backup
        self.data: dict[str, xr.DataArray] = data.copy()
        self.backup_data: dict = {}
        if temper_choose:
            self.backup_data["temper"]: xr.DataArray = self.data["temper"].copy()
        if precip_choose:
            self.backup_data["precip"]: xr.DataArray = self.data["precip"].copy()
        # Get length of the data, remove duplicate data, and slice the data into decades
        if temper_choose:
            len_data: int = len(self.data["temper"]) - 60  # remove duplicate data
        else:
            len_data: int = len(self.data["precip"]) - 60  # remove duplicate data
        # Get the length of the last decade and the number of decades in the data
        rest_size: int = len_data % 10
        decades: int = int(len_data / 10)  # 1961-today (minimal 62 years)
        # Slice the temperature data into decades and remove unnecessary rows and columns
        if temper_choose:
            self.data["temper"]: xr.DataArray = self.data["temper"].isel(time=sum(
                [list(range(len_data - 10 * (i + 1), len_data - 10 * i)) for i in range(decades)], []) + list(
                range(rest_size)))
            self.data["temper"]: xr.DataArray = self.data["temper"].isel(row=slice(1, 1 + 14 * 3, 3),
                                                                         col=slice(2, 2 + 12))
        # Slice the precipitation data into decades and remove unnecessary rows and columns
        if precip_choose:
            self.data["precip"]: xr.DataArray = self.data["precip"].isel(time=sum(
                [list(range(len_data - 10 * (i + 1), len_data - 10 * i)) for i in range(decades)], []) + list(
                range(rest_size)))
            self.data["precip"]: xr.DataArray = self.data["precip"].isel(row=slice(1, 1 + 14 * 3, 3),
                                                                         col=slice(2, 2 + 12))

    def reset_data(func: Callable[..., any]) -> Callable[..., any]:
        """
        This is a decorator function that takes another function as an argument and returns a new function.
        The returned function restores the original data stored in the data dictionary by copying the backup data
        for possible reuse.

        :return: function
        """

        @functools.wraps(func)
        def wrapper(self, *args: any, **kwargs: any) -> func:
            """
            This is a decorator function that wraps around another function passed as an argument.
            it first executes the original function then it restores the original weather data from the backup.

            :param self: The class instance that the decorated function is a method of.
            :param args: Positional arguments passed to the decorated function.
            :param kwargs: Keyword arguments passed to the decorated function.
            :return: The result of the decorated function.
            """
            result: func = func(self, *args, **kwargs)
            if self.temper_choose:
                self.data["temper"]: xr.DataArray = self.backup_data["temper"].copy()
            if self.precip_choose:
                self.data["precip"]: xr.DataArray = self.backup_data["precip"].copy()
            return result

        return wrapper

    @Utils.debug
    def plot_3d(self) -> None:
        """
        Create and display a 3D plot of temperature or precipitation data for the last 5 years.

        :return: None
        """
        self.plot_3d_year_region(predef_years=[year for year in range(2018, 2022 + 1)],
                                 subtitle_part_text="za posledních 5 let",
                                 regions=[region for region in range(0, 14)])

    @Utils.debug
    def plot_3d_year(self) -> None:
        """
        Create and display a 3D plot of temperature or precipitation data for a specific year(s).

        :return: None
        """
        self.plot_3d_year_region(regions=[region for region in range(0, 14)])

    @Utils.debug
    @reset_data
    def plot_3d_year_region(self, predef_years: str = None, subtitle_part_text: str = "", regions: str = None) -> None:
        """
        Create and display a 3D plot of temperature or precipitation data for a specific year(s) and region(s).

        :param predef_years: A list of years to plot, if None it plots all the available years
        :param subtitle_part_text: A substring for the subtitle
        :param regions: A list of regions to display
        :return: None
        """
        if regions is None:
            regions = []
        if predef_years is None:
            predef_years = []

        def create_3d_plot(name: str, years: list[int] = None, regions: list[int] = None,
                           subtitle_part_text: str = subtitle_part_text) -> None:
            """
            This function creates and displays a 3D plot of temperature or precipitation data for a specific year
            or years and region(s).

            :param name: str, the name of the weather variable ("temper" or "precip")
            :param years: list[int], optional, list of years to plot, if None, all years will be plotted
            :param regions: list[int], optional, list of region IDs to plot, if None, all regions will be plotted
            :param subtitle_part_text: str, optional, text to include in the plot subtitle
            :return: None
            """
            if regions is None:
                regions = []
            if years is None:
                years = []

            def on_key_press(event: mpl.backend_bases.KeyEvent) -> None:
                """
                A function that is called when a key is pressed. If the pressed key is 'x', it closes all the figures.

                :param event: The event object that contains information about the key press.
                :return: None
                """
                if event.key.lower() == 'x':
                    plt.close('all')

            def update(val: Union[float, np.float64]) -> None:
                """
                A function that is called when the slider value is changed.
                It sets the alpha (transparency) of the plot surface.

                :param val: The new value of the slider.
                :return: None
                """
                if len(regions) == 1:
                    surf[0].set_alpha(val)
                else:
                    surf.set_alpha(val)

            def scroll_event(event: mpl.backend_bases.MouseEvent) -> None:
                """
                A function that is called when the scroll wheel is used.
                It increases or decreases the alpha (transparency) of the plot surface.

                :param event: The event object that contains information about the scroll event.
                :return: None
                """
                if event.button == 'up':
                    alpha: float = min(1., pos.val + 0.02)
                    pos.set_val(alpha)
                elif event.button == 'down':
                    alpha: float = max(0., pos.val - 0.02)
                    pos.set_val(alpha)

            def rotate_graph(_: mpl.backend_bases.KeyEvent) -> None:
                """
                This is a function that rotates the 3D plot view by changing the azimuth and
                elevation angles of the plot based on keyboard keys.
                The function also stores the time of the last rotation to avoid updating the plot too frequently.

                :param _: Unused argument, required for compatibility with matplotlib callback functions.
                :return: None
                """
                current_time: float = tim.time()
                if current_time - self.last_time > 0.1:  # 200 ms
                    if keyboard.is_pressed('right') and keyboard.is_pressed('up'):
                        ax.view_init(elev=ax.elev - 5, azim=ax.azim - 5)
                        ax.elev -= 5
                        ax.azim -= 5
                    elif keyboard.is_pressed('right') and keyboard.is_pressed('down'):
                        ax.view_init(elev=ax.elev + 5, azim=ax.azim - 5)
                        ax.elev += 5
                        ax.azim -= 5
                    elif keyboard.is_pressed('right'):
                        ax.view_init(elev=ax.elev, azim=ax.azim - 5)
                        ax.azim -= 5
                    elif keyboard.is_pressed('left') and keyboard.is_pressed('up'):
                        ax.view_init(elev=ax.elev - 5, azim=ax.azim + 5)
                        ax.elev -= 5
                        ax.azim += 5
                    elif keyboard.is_pressed('left') and keyboard.is_pressed('down'):
                        ax.view_init(elev=ax.elev + 5, azim=ax.azim + 5)
                        ax.elev += 5
                        ax.azim += 5
                    elif keyboard.is_pressed('left'):
                        ax.view_init(elev=ax.elev, azim=ax.azim + 5)
                        ax.azim += 5
                    elif keyboard.is_pressed('up'):
                        ax.view_init(elev=ax.elev - 5, azim=ax.azim)
                        ax.elev -= 5
                    elif keyboard.is_pressed('down'):
                        ax.view_init(elev=ax.elev + 5, azim=ax.azim)
                        ax.elev += 5
                    fig.canvas.draw()
                    self.last_time = current_time

            def update_text(ax, pos: list[float], text: str, is_temper: bool, bottom: bool) -> None:
                """
                Updates text in the given axis.

                :param ax: The axis to update the text in.
                :param pos: The position of the text.
                :param text: The text to be updated.
                :param is_temper: Whether the text is temperature data or not.
                :param bottom: Whether the text should be placed at the bottom of the graph or not.
                :return: None
                """
                temper_top_backup: mpl.text.Text = None
                # Remove the previous temperature text and backup the top text.
                if ax.texts:
                    if len(ax.texts) > 0:
                        temper_top_backup: mpl.text.Text = ax.texts[-1]
                        ax.texts[-1].remove()
                        # Remove the top text if this is a temperature data and not at the bottom.
                        if is_temper and not bottom and len(ax.texts) > 0:
                            ax.texts[-1].remove()

                # Calculate the position of the text based on the position of the axis.
                x_pos_px: np.float64 = ax.get_position().bounds[0]
                x_pos: Union[np.float64, float] = pos[0] + (-x_pos_px + 0.22375) * 2
                # Add the new text to the axis.
                ax.text2D(x_pos, pos[1], text, transform=ax.transAxes, ha='right', va='top', fontsize=10,
                          color="#006600", alpha=0.8)
                # Add back the temperature text if it was removed before and is not at the bottom.
                if ax.texts and temper_top_backup:
                    if is_temper and not bottom and len(ax.texts) > 0:
                        ax.texts.append(temper_top_backup)

            def plot_stats(fig: mpl.figure.Figure, ax, zz: np.ndarray, name: str,
                           bottom: bool = False) -> None:
                """
                Plots statistics of the given 2D numpy array on the given Matplotlib axis.

                :param fig: Matplotlib Figure to draw on
                :param ax: Matplotlib AxesSubplot to draw on
                :param zz: 2D numpy array to calculate statistics from
                :param name: Name of the data type ('precip', 'temper' or 'elev')
                :param bottom: True if the data is bottom temperature, False otherwise (default: False)
                :return: None
                """
                z_min: np.ndarray = np.nanmin(zz)
                z_max: np.ndarray = np.nanmax(zz)
                z_mean: np.ndarray = np.nanmean(zz)
                if name == "precip" or (name == "temper" and bottom):
                    z_hmean: np.ndarray = hmean(list(itertools.chain.from_iterable(zz)))
                    z_gmean: np.ndarray = gmean(list(itertools.chain.from_iterable(zz)))
                else:
                    z_hmean: float = 0
                    z_gmean: float = 0
                z_q1: float = np.nanpercentile(zz, 25)
                z_median: np.ndarray = np.nanmedian(zz)
                z_q3: float = np.nanpercentile(zz, 75)
                z_var: np.ndarray = np.nanvar(zz)
                z_std: np.ndarray = np.nanstd(zz)
                z_skew: np.ndarray = skew(np.array(list(itertools.chain.from_iterable(zz))))
                z_kurtosis: float = kurtosis(list(itertools.chain.from_iterable(zz)))

                unit: str = "[K]   " if bottom else "[°C] " if name == 'temper' else "[mm] "
                min_str: str = f"Minimum: {z_min:>8.2f} {unit}\n"
                max_str: str = f"Maximum: {z_max:>8.2f} {unit}\n"
                mean_str: str = f"Průměr: {z_mean:>8.2f} {unit}\n"
                if name == "precip" or (name == "temper" and bottom):
                    hmean_str: str = f"Harm. průměr: {z_hmean:>8.2f} {unit}\n"
                    gmean_str: str = f"Geom. průměr: {z_gmean:>8.2f} {unit}\n"
                else:
                    gmean_str: str = ""
                    hmean_str: str = ""
                q1_str: str = f"1. kvartil: {z_q1:>8.2f} {unit}\n"
                median_str: str = f"Medián: {z_median:>8.2f} {unit}\n"
                q3_str: str = f"3. kvartil: {z_q3:>8.2f} {unit}\n"
                idx: int = 4 if bottom else 2
                var_str: str = f"Rozptyl: {z_var:>8.2f} {unit[:-idx] + '²' + unit[-idx:-1]}\n"
                std_str: str = f"Směr. odchylka: {z_std:>8.2f} {unit}\n"
                space_size: int = 4 if name == 'temper' else 6
                skew_str: str = f"Koef. šikmosti: {z_skew:>8.2f} [-]{' ' * space_size}\n"
                kurtosis_str: str = f"Koef. špičatosti: {z_kurtosis:>8.2f} [-]{' ' * space_size}\n"

                pos: list[float] = [0.10, 0.95 if not bottom else 0.5]
                text: str = min_str + max_str + mean_str + hmean_str + gmean_str + q1_str + median_str + q3_str
                text += var_str + std_str + skew_str + kurtosis_str
                is_temper: bool = True if name == "temper" else False
                fig.canvas.mpl_connect('resize_event', lambda event: update_text(ax, pos, text, is_temper, bottom))

            # Make a copy of the weather data
            weather_data: xr.DataArray = self.data[name].copy()
            # Select the desired years and concatenate them along the 'col' dimension
            weather_data: xr.DataArray = xr.concat([weather_data.isel(time=year - 1961) for year in years], dim='col')
            # Select the desired regions
            weather_data: xr.DataArray = weather_data.sel(row=regions)

            # Create the mesh grid for the plot
            x: np.ndarray = weather_data.coords['col'].values  # months
            y: np.ndarray = weather_data.coords['row'].values  # regions
            xx: np.ndarray
            yy: np.ndarray
            xx, yy = np.meshgrid(x, y)

            # Replace comma with dot in the data and convert it to float
            z: np.ndarray = weather_data.values
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    z[i][j] = z[i][j].replace(',', '.')
            zz: np.ndarray = z.astype(float)

            # Create the figure and subplot with 3D projection
            fig: mpl.figure.Figure
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})

            # Create the title of the plot
            years_str: str = ", ".join(map(str, years[:10])) + (", ..." if len(years) > 10 else "")
            if not subtitle_part_text:
                subtitle_part_text = "3D graf {} za roky: {}".format(
                    "teplot" if name == 'temper' else "srážek", years_str)
            else:
                subtitle_part_text = "3D graf {} {}".format(
                    "teplot" if name == 'temper' else "srážek", subtitle_part_text)
            fig.suptitle(subtitle_part_text)

            # Create the surface plot
            surf: mpl.collections.PolyCollection
            if len(regions) == 1:
                surf = ax.plot(xx[0], zz[0], zdir='y', alpha=1)
            else:
                surf = ax.plot_surface(xx, yy, zz, alpha=1)

            # Set the ticks and labels for the x-axis
            ax.set_xticks(x)
            month_labels: list[str] = [f"{month} {year}" for year in years for month in self.months]
            month_labels: list[str] = ['' if (i + 1) % (len(month_labels) // 12) != 0 else month_labels[i]
                                       for i in range(len(month_labels))]
            ax.set_xticklabels(month_labels, rotation=60, ha='right')

            # Set the ticks and labels for the y-axis
            ax.set_yticks(y)
            sel_regions: list[str] = [self.regions[i] for i in regions]
            ax.set_yticklabels(sel_regions, rotation=60, ha='right')

            # Set the ticks and labels for the z-axis
            z_min: np.float64 = min([min(elem) for elem in zz])
            z_max: np.float64 = max([max(elem) for elem in zz])
            step: Union[np.float64, float] = (z_max - z_min) / 10.0
            ax.set_zticks(np.arange(np.floor(z_min), np.ceil(z_max) + step, step))
            ax.set_xlabel('Měsíce [-]', color='blue', va='bottom', labelpad=35)
            ax.set_ylabel('Kraje [-]', color='blue', labelpad=35)
            ax.set_zlabel('Teplota [°C]' if name == 'temper' else 'Srážky [mm]', color='blue')

            # Calling the function to plot statistics
            plot_stats(fig, ax, zz, name)
            # Statistics in Kelvin are also plotted for temperatures.
            if name == "temper":
                zz += 273.15
                zz[zz <= 0] += 1e-6
                plot_stats(fig, ax, zz, name, True)

            # Setting the position of the slider and its characteristics
            pos_slider: mpl.axes.Axes = plt.axes([0.2, 0.9, 0.65, 0.03], facecolor='lightgray')
            pos: mpl.widgets.Slider = plt.Slider(pos_slider, 'Alfa', 0, 1, valinit=1)
            mpl.rcParams.update({'keymap.forward': ['d', 'D'], 'keymap.back': ['a', 'A'], 'keymap.save': ['s', 'S']})
            pos.on_changed(update)
            fig.canvas.mpl_connect('key_press_event', rotate_graph)
            fig.canvas.mpl_connect('scroll_event', scroll_event)
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            # Display the final figure
            fig.show()

        def control_years(min_year: int, max_year: int, temp_time_len: int) -> bool:
            """
            Checks if the given range of years is valid or not.

            :param min_year: minimum year in the range
            :param max_year: maximum year in the range
            :param temp_time_len: length of the temperature data
            :return: boolean indicating if the years are valid or not
            """
            if min_year < 1961 or max_year > 1961 + temp_time_len - 1:
                print("Zadané roky nejsou v rozsahu 1961-" + str(1961 + temp_time_len - 1))
                return False
            return True

        def control_regions(regions: list[int]) -> bool:
            """
            Controls whether the regions entered by the user are valid or not.

            :param regions: A list of integers representing regions.
            :return: A boolean indicating whether the regions are valid or not.
            """
            if min(regions) < 0 or max(regions) > 13:
                print("Zadané kraje nejsou v rozsahu 0-13")
                return False
            return True

        # Ask the user to input the years
        years: list[int]
        if years := UserInterface.input_loop("Zadejte rok(y)", year=True) if not subtitle_part_text else predef_years:
            # Display the regions
            print("Kraje:") if not regions else regions
            print("\n".join([f"{i}) {region}" for i, region in enumerate(self.regions)])) if not regions else regions
            # Ask the user to input the regions
            regions: list[int]
            if not (regions := UserInterface.input_loop("Zadejte kraj(e)", region=True) if not regions else regions):
                return

            # Validate the input regions
            if not control_regions(regions):
                return

            # Set the minimum and maximum years based on user input
            min_year: int = min(years)
            max_year: int = max(years)

            # Create a 3D plot for temperature if selected by user
            if self.temper_choose:
                temp_data: xr.DataArray = self.data['temper'].copy()
                temp_time_len: int = len(temp_data.coords['time'])

                # Validate the input years for temperature
                if not control_years(min_year, max_year, temp_time_len):
                    return
                create_3d_plot("temper", years, regions)

            # Create a 3D plot for precipitation if selected by user
            if self.precip_choose:
                prec_data: xr.DataArray = self.data['precip'].copy()
                prec_time_len: int = len(prec_data.coords['time'])

                # Validate the input years for precipitation
                if not control_years(min_year, max_year, prec_time_len):
                    return
                create_3d_plot("precip", years, regions)

            # Show the plot to the user
            plt.show()
        else:
            return

    @Utils.debug
    @reset_data
    def plot_2d_year_region(self) -> None:
        """
        Create and display a 2D plot of temperature or precipitation data for a specific year(s) and region(s).

        :return: None
        """

        def create_2d_plot(name: str, years: list = None, regions: list = None) -> None:
            """
            This function creates and displays a 2D plot of temperature or precipitation data for a specific year
            or years and region(s).

            :param name: str, the name of the weather variable ("temper" or "precip")
            :param years: list[int], optional, list of years to plot, if None, all years will be plotted
            :param regions: list[int], optional, list of region IDs to plot, if None, all regions will be plotted
            :return: None
            """
            def on_key_press(event: mpl.backend_bases.KeyEvent) -> None:
                """
                A function that is called when a key is pressed. If the pressed key is 'x', it closes all the figures.

                :param event: The event object that contains information about the key press.
                :return: None
                """
                if event.key.lower() == 'x':
                    plt.close('all')

            def update(val: Union[float, np.float64]) -> None:
                """
                A function that is called when the slider value is changed.
                It sets the alpha (transparency) of the plot.

                :param val: The new value of the slider.
                :return: None
                """
                for line in lines:
                    line.set_alpha(val)
                plt.draw()

            def scroll_event(event: mpl.backend_bases.MouseEvent) -> None:
                """
                A function that is called when the scroll wheel is used.
                It increases or decreases the alpha (transparency) of the plot surface.

                :param event: The event object that contains information about the scroll event.
                :return: None
                """
                if event.button == 'up':
                    alpha = min(1., pos.val + 0.02)
                    pos.set_val(alpha)
                elif event.button == 'down':
                    alpha = max(0., pos.val - 0.02)
                    pos.set_val(alpha)

            def update_text(ax, pos: list[float], text: str, is_temper: bool, bottom: bool) -> None:
                """
                Updates text in the given axis.

                :param ax: The axis to update the text in.
                :param pos: The position of the text.
                :param text: The text to be updated.
                :param is_temper: Whether the text is temperature data or not.
                :param bottom: Whether the text should be placed at the bottom of the graph or not.
                :return: None
                """
                temper_top_backup = None
                if ax.texts:
                    if len(ax.texts) > 0:
                        temper_top_backup = ax.texts[-1]
                        ax.texts[-1].remove()
                        if is_temper and not bottom and len(ax.texts) > 0:
                            ax.texts[-1].remove()

                x_pos_px = ax.get_position().bounds[0]
                x_pos = pos[0] + (-x_pos_px + 0.22375) * 2
                ax.text(x_pos, pos[1], text, transform=ax.transAxes, ha='right', va='top', fontsize=10,
                        color="#006600", alpha=0.8)
                if ax.texts and temper_top_backup:
                    if is_temper and not bottom and len(ax.texts) > 0:
                        ax.texts.append(temper_top_backup)

            def plot_stats(fig: mpl.figure.Figure, ax, zz: np.ndarray, name: str, right: bool = False) -> None:
                z_min: np.ndarray = np.nanmin(zz)
                z_max: np.ndarray = np.nanmax(zz)
                z_mean: np.ndarray = np.nanmean(zz)
                if name == "precip" or (name == "temper" and right):
                    z_hmean: np.ndarray = hmean(list(itertools.chain.from_iterable(zz)))
                    z_gmean: np.ndarray = gmean(list(itertools.chain.from_iterable(zz)))
                else:
                    z_hmean: float = 0
                    z_gmean: float = 0
                z_q1: float = np.nanpercentile(zz, 25)
                z_median: np.ndarray = np.nanmedian(zz)
                z_q3: float = np.nanpercentile(zz, 75)
                z_var: np.ndarray = np.nanvar(zz)
                z_std: np.ndarray = np.nanstd(zz)
                z_skew: np.ndarray = skew(np.array(list(itertools.chain.from_iterable(zz))))
                z_kurtosis: float = kurtosis(list(itertools.chain.from_iterable(zz)))

                unit: str = "[K]   " if right else "[°C] " if name == 'temper' else "[mm] "
                min_str: str = f"Minimum: {z_min:>8.2f} {unit}\n"
                max_str: str = f"Maximum: {z_max:>8.2f} {unit}\n"
                mean_str: str = f"Průměr: {z_mean:>8.2f} {unit}\n"
                if name == "precip" or (name == "temper" and right):
                    hmean_str: str = f"Harm. průměr: {z_hmean:>8.2f} {unit}\n"
                    gmean_str: str = f"Geom. průměr: {z_gmean:>8.2f} {unit}\n"
                else:
                    gmean_str: str = ""
                    hmean_str: str = ""
                q1_str: str = f"1. kvartil: {z_q1:>8.2f} {unit}\n"
                median_str: str = f"Medián: {z_median:>8.2f} {unit}\n"
                q3_str: str = f"3. kvartil: {z_q3:>8.2f} {unit}\n"
                idx: int = 4 if right else 2
                var_str: str = f"Rozptyl: {z_var:>8.2f} {unit[:-idx] + '²' + unit[-idx:-1]}\n"
                std_str: str = f"Směr. odchylka: {z_std:>8.2f} {unit}\n"
                space_size: int = 4 if name == 'temper' else 6
                skew_str: str = f"Koef. šikmosti: {z_skew:>8.2f} [-]{' ' * space_size}\n"
                kurtosis_str: str = f"Koef. špičatosti: {z_kurtosis:>8.2f} [-]{' ' * space_size}\n"

                pos: list[float] = [0.15 if not right else 0.75, 0.95]
                text: str = min_str + max_str + mean_str + hmean_str + gmean_str + q1_str + median_str + q3_str
                text += var_str + std_str + skew_str + kurtosis_str
                is_temper: bool = True if name == "temper" else False
                fig.canvas.mpl_connect('resize_event', lambda event: update_text(ax, pos, text, is_temper, right))

            # Make a copy of the weather data
            weather_data = self.data[name].copy()
            # Select the desired years and concatenate them along the 'col' dimension
            weather_data = xr.concat([weather_data.isel(time=year - 1961) for year in years], dim='col')
            # Select the desired regions
            weather_data = weather_data.sel(row=regions)

            x: np.ndarray = weather_data.coords['col'].values  # months
            y: np.ndarray = weather_data.values
            z: np.ndarray = weather_data.coords['row'].values  # regions

            # Replace comma with dot in the data and convert it to float
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    y[i][j] = y[i][j].replace(',', '.')
            yy: np.ndarray = y.astype(float)

            fig: mpl.figure.Figure = plt.figure(figsize=(10, 5))
            years_str: str = ", ".join(map(str, years[:10])) + (", ..." if len(years) > 10 else "")
            subtitle_part_text: str = "2D graf {} za roky: {}".format(
                "teplot" if name == 'temper' else "srážek", years_str)
            fig.suptitle(subtitle_part_text)

            sel_regions: list[str] = [self.regions[i] for i in regions]
            lines: list[mpl.lines.Line2D] = []
            for i in range(len(z)):
                lines += plt.plot(x, yy[i], label=sel_regions[i], alpha=1)
            legend: mpl.legend.Legend = plt.legend(title='Kraje:')
            legend.set_bbox_to_anchor((1, 1))

            ax = plt.gca()
            ax.set_xticks(x)
            month_labels: list[str] = [f"{month} {year}" for year in years for month in self.months]
            month_labels: list[str] = ['' if (i + 1) % (len(month_labels) // 12) != 0 else month_labels[i]
                                       for i in range(len(month_labels))]
            ax.set_xticklabels(month_labels, rotation=20, ha='right')

            # Set the ticks and labels for the y-axis
            y_min: np.float64 = min([min(elem) for elem in yy])
            y_max: np.float64 = max([max(elem) for elem in yy])
            step: Union[np.float64, float] = (y_max - y_min) / 10.0
            ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + step, step))
            fig.subplots_adjust(bottom=0.2)  # Posunutí celého grafu nahoru o 0.1
            fig.subplots_adjust(right=0.75)
            ax.set_xlabel('Měsíce [-]', color='blue', va='bottom', labelpad=35)
            ax.xaxis.set_label_coords(0.5, -0.25)  # Nastavení nových hodnot pro souřadnice x a y
            ax.set_ylabel('Teplota [°C]' if name == 'temper' else 'Srážky [mm]', color='blue', labelpad=35)
            ax.yaxis.set_label_coords(-0.1, 0.5)  # Nastavení nových hodnot pro souřadnice x a y

            # Calling the function to plot statistics
            plot_stats(fig, ax, yy, name)
            # Statistics in Kelvin are also plotted for temperatures.
            if name == "temper":
                yy += 273.15
                yy[yy <= 0] += 1e-6
                plot_stats(fig, ax, yy, name, True)

            # Setting the position of the slider and its characteristics
            pos_slider: mpl.axes.Axes = plt.axes([0.2, 0.9, 0.65, 0.03], facecolor='lightgray')
            pos: mpl.widgets.Slider = plt.Slider(pos_slider, 'Alfa', 0, 1, valinit=1)
            pos.on_changed(update)
            mpl.rcParams.update({'keymap.forward': ['d', 'D'], 'keymap.back': ['a', 'A'], 'keymap.save': ['s', 'S']})
            fig.canvas.mpl_connect('scroll_event', scroll_event)
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            # Display the final figure
            fig.show()

        def control_years(min_year: int, max_year: int, temp_time_len: int) -> bool:
            """
            Checks if the given range of years is valid or not.

            :param min_year: minimum year in the range
            :param max_year: maximum year in the range
            :param temp_time_len: length of the temperature data
            :return: boolean indicating if the years are valid or not
            """
            if min_year < 1961 or max_year > 1961 + temp_time_len - 1:
                print("Zadané roky nejsou v rozsahu 1961-" + str(1961 + temp_time_len - 1))
                return False
            return True

        def control_regions(regions: list[int]) -> bool:
            """
            Controls whether the regions entered by the user are valid or not.

            :param regions: A list of integers representing regions.
            :return: A boolean indicating whether the regions are valid or not.
            """
            if min(regions) < 0 or max(regions) > 13:
                print("Zadané kraje nejsou v rozsahu 0-13")
                return False
            return True

        # Ask the user to input the years
        years: list[int]
        if years := UserInterface.input_loop("Zadejte rok(y)", year=True):
            # Display the regions
            print("Kraje:")
            print("\n".join([f"{i}) {region}" for i, region in enumerate(self.regions)]))
            # Ask the user to input the regions
            regions: list[int]
            if not (regions := UserInterface.input_loop("Zadejte kraj(e)", region=True)):
                return

            # Validate the input regions
            if not control_regions(regions):
                return

            # Set the minimum and maximum years based on user input
            min_year: int = min(years)
            max_year: int = max(years)

            # Create a 2D plot for temperature if selected by user
            if self.temper_choose:
                temp_data: xr.DataArray = self.data['temper'].copy()
                temp_time_len: int = len(temp_data.coords['time'])

                # Validate the input years for temperature
                if not control_years(min_year, max_year, temp_time_len):
                    return
                create_2d_plot("temper", years, regions)

            # Create a 2D plot for precipitation if selected by user
            if self.precip_choose:
                prec_data: xr.DataArray = self.data['precip'].copy()
                prec_time_len: int = len(prec_data.coords['time'])

                # Validate the input years for temperature
                if not control_years(min_year, max_year, prec_time_len):
                    return
                create_2d_plot("precip", years, regions)
            # Show the plot to the user
            plt.show()
        else:
            return

    @Utils.debug
    def plot_3d_hist_dct(self) -> None:
        """
        Create and display a 3D histogram of discrete cosine transform (DCT) coefficients of temperature or
        precipitation data for the last 5 years.

        :return: None
        """
        self.plot_3d_hist_dct_year_region(predef_years=[year for year in range(2018, 2022 + 1)],
                                          subtitle_part_text="za posledních 5 let",
                                          regions=[region for region in range(0, 14)])

    @Utils.debug
    @reset_data
    def plot_3d_hist_dct_year_region(self, predef_years: str = None, subtitle_part_text: str = "",
                                     regions: str = None) -> None:
        """
        Create and display a 3D histogram of DCT coefficients of temperature or precipitation data for a specific year
        or years and region(s).

        :param predef_years: Optional string of predefined years to use for the plot.
        :param subtitle_part_text:  Optional substring to add to the subtitle of the plot.
        :param regions: Optional string of predefined regions to use for the plot.
        :return: None
        """
        if regions is None:
            regions = []
        if predef_years is None:
            predef_years = []

        def create_3d_plot(name: str, years: list = None, regions: list = None,
                           subtitle_part_text: str = subtitle_part_text) -> None:
            """
            This function creates and displays a 3D plot of temperature or precipitation DCT data for a specific year
            or years and region(s).

            :param name: str, the name of the weather variable ("temper" or "precip")
            :param years: list[int], optional, list of years to plot, if None, all years will be plotted
            :param regions: list[int], optional, list of region IDs to plot, if None, all regions will be plotted
            :param subtitle_part_text: str, optional, text to include in the plot subtitle
            :return: None
            """
            def on_key_press(event: mpl.backend_bases.KeyEvent) -> None:
                """
                A function that is called when a key is pressed. If the pressed key is 'x', it closes all the figures.

                :param event: The event object that contains information about the key press.
                :return: None
                """
                if event.key.lower() == 'x':
                    plt.close('all')

            def update(val: Union[float, np.float64]) -> None:
                """
                A function that is called when the slider value is changed.
                It sets the alpha (transparency) of the plot surface.

                :param val: The new value of the slider.
                :return: None
                """
                surf.set_alpha(val)

            def scroll_event(event: mpl.backend_bases.MouseEvent) -> None:
                """
                A function that is called when the scroll wheel is used.
                It increases or decreases the alpha (transparency) of the plot surface.

                :param event: The event object that contains information about the scroll event.
                :return: None
                """
                if event.button == 'up':
                    alpha = min(1., pos.val + 0.02)
                    pos.set_val(alpha)
                elif event.button == 'down':
                    alpha = max(0., pos.val - 0.02)
                    pos.set_val(alpha)

            def rotate_graph(_: mpl.backend_bases.KeyEvent) -> None:
                """
                This is a function that rotates the 3D plot view by changing the azimuth and
                elevation angles of the plot based on keyboard keys.
                The function also stores the time of the last rotation to avoid updating the plot too frequently.

                :param _: Unused argument, required for compatibility with matplotlib callback functions.
                :return: None
                """
                current_time: float = tim.time()
                if current_time - self.last_time > 0.1:  # 200 ms
                    if keyboard.is_pressed('right') and keyboard.is_pressed('up'):
                        ax.view_init(elev=ax.elev - 5, azim=ax.azim - 5)
                        ax.elev -= 5
                        ax.azim -= 5
                    elif keyboard.is_pressed('right') and keyboard.is_pressed('down'):
                        ax.view_init(elev=ax.elev + 5, azim=ax.azim - 5)
                        ax.elev += 5
                        ax.azim -= 5
                    elif keyboard.is_pressed('right'):
                        ax.view_init(elev=ax.elev, azim=ax.azim - 5)
                        ax.azim -= 5
                    elif keyboard.is_pressed('left') and keyboard.is_pressed('up'):
                        ax.view_init(elev=ax.elev - 5, azim=ax.azim + 5)
                        ax.elev -= 5
                        ax.azim += 5
                    elif keyboard.is_pressed('left') and keyboard.is_pressed('down'):
                        ax.view_init(elev=ax.elev + 5, azim=ax.azim + 5)
                        ax.elev += 5
                        ax.azim += 5
                    elif keyboard.is_pressed('left'):
                        ax.view_init(elev=ax.elev, azim=ax.azim + 5)
                        ax.azim += 5
                    elif keyboard.is_pressed('up'):
                        ax.view_init(elev=ax.elev - 5, azim=ax.azim)
                        ax.elev -= 5
                    elif keyboard.is_pressed('down'):
                        ax.view_init(elev=ax.elev + 5, azim=ax.azim)
                        ax.elev += 5
                    fig.canvas.draw()
                    self.last_time = current_time

            # Make a copy of the weather data
            weather_data: xr.DataArray = self.data[name].copy()
            # Select the desired years and concatenate them along the 'col' dimension
            weather_data: xr.DataArray = xr.concat([weather_data.isel(time=year - 1961) for year in years], dim='col')
            # Select the desired regions
            weather_data: xr.DataArray = weather_data.sel(row=regions)

            z_pre: np.ndarray = weather_data.values
            # Replace comma with dot in the data and convert it to float
            for i in range(z_pre.shape[0]):
                for j in range(z_pre.shape[1]):
                    z_pre[i][j] = z_pre[i][j].replace(',', '.')
            z_pre: np.ndarray = z_pre.astype(float)

            # Calculate the Discrete Cosine Transform of the data for each region
            dct_region: np.ndarray = np.zeros_like(z_pre)
            for i in range(len(regions)):
                dct_region[i, :] = dct(z_pre[i, :], type=2, norm='ortho')

            # Create the 3D plot
            fig: mpl.figure.Figure
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
            _x: np.ndarray = np.arange(len(dct_region[0]))
            _y: np.ndarray = np.arange(len(dct_region))
            _xx: np.ndarray
            _yy: np.ndarray
            _xx, _yy = np.meshgrid(_x, _y)
            x: np.ndarray
            y: np.ndarray
            x, y = _xx.ravel(), _yy.ravel()
            bottom: np.ndarray = np.zeros_like(x)
            width: int = 1
            depth: int = 1

            surf: mpl.mpl_toolkits.mplot3d.art3d.Poly3DCollection = ax.bar3d(x, y, bottom, width, depth,
                                                                             np.abs(dct_region).ravel(), shade=True)
            # Add subtitle to the plot
            years_str: str = ", ".join(map(str, years[:10])) + (", ..." if len(years) > 10 else "")
            subtitle_part_text: str
            if not subtitle_part_text:
                subtitle_part_text = "3D graf spektra {} za roky: {}".format(
                    "teplot" if name == 'temper' else "srážek", years_str)
            else:
                subtitle_part_text = "3D graf spektra {} {}".format(
                    "teplot" if name == 'temper' else "srážek", subtitle_part_text)
            fig.suptitle(subtitle_part_text)

            # Configure the x-axis ticks and labels
            freq: np.ndarray = np.linspace(_x.min(), _x.max(), 10)
            ax.set_xticks(freq)
            ax.set_xticklabels(np.round(freq / (2 * len(dct_region[0])), decimals=2), rotation=80, ha='right')

            # Configure the y-axis ticks and labels
            ax.set_yticks(_y)
            sel_regions: list[str] = [self.regions[i] for i in regions]
            ax.set_yticklabels(sel_regions, rotation=60, ha='right')

            # Add labels to the axes
            ax.set_xlabel('Frekvence [1/měsíc]', color='blue', va='bottom', labelpad=35)
            ax.set_ylabel('Kraje [-]', color='blue', labelpad=35)
            ax.set_zlabel('Absolutní hodnota spektra ' + ('teploty [°C]' if name == 'temper' else 'srážek [mm]'),
                          color='blue')

            pos_slider: mpl.axes.Axes = plt.axes([0.2, 0.9, 0.65, 0.03], facecolor='lightgray')
            pos: mpl.widgets.Slider = plt.Slider(pos_slider, 'Alfa', 0, 1, valinit=1)
            mpl.rcParams.update({'keymap.forward': ['d', 'D'], 'keymap.back': ['a', 'A'], 'keymap.save': ['s', 'S']})
            pos.on_changed(update)
            fig.canvas.mpl_connect('key_press_event', rotate_graph)
            fig.canvas.mpl_connect('scroll_event', scroll_event)
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            # display the figure
            fig.show()

        def control_years(min_year: int, max_year: int, temp_time_len: int) -> bool:
            """
            Checks if the given range of years is valid or not.

            :param min_year: minimum year in the range
            :param max_year: maximum year in the range
            :param temp_time_len: length of the temperature data
            :return: boolean indicating if the years are valid or not
            """
            if min_year < 1961 or max_year > 1961 + temp_time_len - 1:
                print("Zadané roky nejsou v rozsahu 1961-" + str(1961 + temp_time_len - 1))
                return False
            return True

        def control_regions(regions: list[int]) -> bool:
            """
            Controls whether the regions entered by the user are valid or not.

            :param regions: A list of integers representing regions.
            :return: A boolean indicating whether the regions are valid or not.
            """
            if min(regions) < 0 or max(regions) > 13:
                print("Zadané kraje nejsou v rozsahu 0-13")
                return False
            return True

        # Ask the user to input the years
        years: list[int]
        if years := UserInterface.input_loop("Zadejte rok(y)", year=True) if not subtitle_part_text else predef_years:
            # Display the regions
            print("Kraje:") if not regions else regions
            print("\n".join([f"{i}) {region}" for i, region in enumerate(self.regions)])) if not regions else regions
            # Ask the user to input the regions
            regions: list[int]
            if not (regions := UserInterface.input_loop("Zadejte kraj(e)", region=True) if not regions else regions):
                return

            # Validate the input regions
            if not control_regions(regions):
                return

            # Set the minimum and maximum years based on user input
            min_year: int = min(years)
            max_year: int = max(years)

            # Create a 3D plot for temperature if selected by user
            if self.temper_choose:
                temp_data: xr.DataArray = self.data['temper'].copy()
                temp_time_len: int = len(temp_data.coords['time'])

                # Validate the input years for temperature
                if not control_years(min_year, max_year, temp_time_len):
                    return
                create_3d_plot("temper", years, regions)

            # Create a 3D plot for precipitation if selected by user
            if self.precip_choose:
                prec_data: xr.DataArray = self.data['precip'].copy()
                prec_time_len: int = len(prec_data.coords['time'])

                # Validate the input years for precipitation
                if not control_years(min_year, max_year, prec_time_len):
                    return
                create_3d_plot("precip", years, regions)
            # Show the plot to the user
            plt.show()
        else:
            return

    @Utils.debug
    @reset_data
    def plot_2d_hist_dct_year_region(self) -> None:
        """
        Create and display a 2D histogram of DCT coefficients of temperature or precipitation data for a specific year
        or years and region(s).

        :return: None
        """

        def create_2d_plot(name: str, years: list = None, regions: list = None) -> None:
            """
            This function creates and displays a 2D plot of temperature or precipitation DCT data for a specific year
            or years and region(s).

            :param name: str, the name of the weather variable ("temper" or "precip")
            :param years: list[int], optional, list of years to plot, if None, all years will be plotted
            :param regions: list[int], optional, list of region IDs to plot, if None, all regions will be plotted
            :return: None
            """
            def on_key_press(event: mpl.backend_bases.KeyEvent) -> None:
                """
                A function that is called when a key is pressed. If the pressed key is 'x', it closes all the figures.

                :param event: The event object that contains information about the key press.
                :return: None
                """
                if event.key.lower() == 'x':
                    plt.close('all')

            def update(val: Union[float, np.float64]) -> None:
                """
                A function that is called when the slider value is changed.
                It sets the alpha (transparency) of the plot surface.

                :param val: The new value of the slider.
                :return: None
                """
                for line in lines:
                    line.set_alpha(val)
                plt.draw()

            def scroll_event(event: mpl.backend_bases.MouseEvent) -> None:
                """
                A function that is called when the scroll wheel is used.
                It increases or decreases the alpha (transparency) of the plot surface.

                :param event: The event object that contains information about the scroll event.
                :return: None
                """
                if event.button == 'up':
                    alpha = min(1., pos.val + 0.02)
                    pos.set_val(alpha)
                elif event.button == 'down':
                    alpha = max(0., pos.val - 0.02)
                    pos.set_val(alpha)

            # Make a copy of the weather data
            weather_data: xr.DataArray = self.data[name].copy()
            # Select the desired years and concatenate them along the 'col' dimension
            weather_data: xr.DataArray = xr.concat([weather_data.isel(time=year - 1961) for year in years], dim='col')
            # Select the desired regions
            weather_data: xr.DataArray = weather_data.sel(row=regions)

            z_pre: np.ndarray = weather_data.values
            for i in range(z_pre.shape[0]):
                for j in range(z_pre.shape[1]):
                    z_pre[i][j] = z_pre[i][j].replace(',', '.')
            z_pre: np.ndarray = z_pre.astype(float)

            # Calculate the Discrete Cosine Transform of the data for each region
            dct_region: np.ndarray = np.zeros_like(z_pre)
            for i in range(len(regions)):
                dct_region[i, :] = dct(z_pre[i, :], type=2, norm='ortho')

            fig: mpl.figure.Figure = plt.figure(figsize=(10, 5))
            years_str: str = ", ".join(map(str, years[:10])) + (", ..." if len(years) > 10 else "")
            subtitle_part_text: str = "2D graf spektra {} za roky: {}".format(
                "teplot" if name == 'temper' else "srážek", years_str)
            fig.suptitle(subtitle_part_text)

            _x: np.ndarray = np.arange(len(dct_region[0]))
            _y: np.ndarray = np.arange(len(dct_region))

            ax = plt.gca()
            n_signals: int = dct_region.shape[0]
            width: float = 0.95 / n_signals
            offset: float = (n_signals - 1) * width / 2
            sel_regions: list[str] = [self.regions[i] for i in regions]
            lines: list = []
            for i in range(n_signals):
                lines += ax.bar(_x + i * width - offset, np.abs(dct_region[i]).ravel(), width, label=sel_regions[i],
                                alpha=1)

            legend: mpl.legend.Legend = plt.legend(title='Kraje:')
            legend.set_bbox_to_anchor((1, 1))
            plt.grid(True)  # Povolení zobrazení mřížky

            freq: np.ndarray = np.linspace(_x.min(), _x.max(), 10)
            ax.set_xticks(freq)
            ax.set_xticklabels(np.round(freq / (2 * len(dct_region[0])), decimals=2), rotation=20, ha='right')

            y_min: Union[np.float64, float] = min([min(elem) for elem in np.abs(dct_region)])
            y_max: Union[np.float64, float] = max([max(elem) for elem in np.abs(dct_region)])
            step: Union[np.float64, float] = (y_max - y_min) / 10.0
            ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + step, step))

            # Add labels to the axes
            ax.set_xlabel('Frekvence [1/měsíc]', color='blue', va='bottom', labelpad=25)
            ax.set_ylabel('Absolutní hodnota spektra ' + ('teploty [°C]' if name == 'temper' else 'srážek [mm]'),
                          color='blue')

            fig.subplots_adjust(bottom=0.2)
            fig.subplots_adjust(right=0.75)

            pos_slider: mpl.axes.Axes = plt.axes([0.2, 0.9, 0.65, 0.03], facecolor='lightgray')
            pos: mpl.widgets.Slider = plt.Slider(pos_slider, 'Alfa', 0, 1, valinit=1)
            pos.on_changed(update)
            mpl.rcParams.update({'keymap.forward': ['d', 'D'], 'keymap.back': ['a', 'A'], 'keymap.save': ['s', 'S']})
            fig.canvas.mpl_connect('scroll_event', scroll_event)
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            # display the figure
            fig.show()

        def control_years(min_year: int, max_year: int, temp_time_len: int) -> bool:
            """
            Checks if the given range of years is valid or not.

            :param min_year: minimum year in the range
            :param max_year: maximum year in the range
            :param temp_time_len: length of the temperature data
            :return: boolean indicating if the years are valid or not
            """
            if min_year < 1961 or max_year > 1961 + temp_time_len - 1:
                print("Zadané roky nejsou v rozsahu 1961-" + str(1961 + temp_time_len - 1))
                return False
            return True

        def control_regions(regions: list[int]) -> bool:
            """
            Controls whether the regions entered by the user are valid or not.

            :param regions: A list of integers representing regions.
            :return: A boolean indicating whether the regions are valid or not.
            """
            if min(regions) < 0 or max(regions) > 13:
                print("Zadané kraje nejsou v rozsahu 0-13")
                return False
            return True

        # Ask the user to input the years
        years: list[int]
        if years := UserInterface.input_loop("Zadejte rok(y)", year=True):
            # Display the regions
            print("Kraje:")
            print("\n".join([f"{i}) {region}" for i, region in enumerate(self.regions)]))
            # Ask the user to input the regions
            regions: list[int]
            if not (regions := UserInterface.input_loop("Zadejte kraj(e)", region=True)):
                return

            # Validate the input regions
            if not control_regions(regions):
                return

            # Set the minimum and maximum years based on user input
            min_year: int = min(years)
            max_year: int = max(years)

            # Create a 2D plot for temperature if selected by user
            if self.temper_choose:
                temp_data: xr.DataArray = self.data['temper'].copy()
                temp_time_len: int = len(temp_data.coords['time'])

                # Validate the input years for temperature
                if not control_years(min_year, max_year, temp_time_len):
                    return
                create_2d_plot("temper", years, regions)

            # Create a 2D plot for precipitation if selected by user
            if self.precip_choose:
                prec_data: xr.DataArray = self.data['precip'].copy()
                prec_time_len: int = len(prec_data.coords['time'])

                # Validate the input years for precipitation
                if not control_years(min_year, max_year, prec_time_len):
                    return
                create_2d_plot("precip", years, regions)
            # Show the plot to the user
            plt.show()
        else:
            return

    @Utils.debug
    @reset_data
    def plot_boxplot_year_region(self) -> None:
        """
        Create and display boxplots of temperature or precipitation data for a specific year(s) and region(s).

        :return: None
        """
        def create_2d_plot(name: str, years: list = None, regions: list = None):
            """
            This function creates and displays a 2D boxplots of temperature or precipitation data for a specific year(s)
            and region(s).

            :param name: str, the name of the weather variable ("temper" or "precip")
            :param years: list[int], optional, list of years to plot, if None, all years will be plotted
            :param regions: list[int], optional, list of region IDs to plot, if None, all regions will be plotted
            :return: None
            """
            def on_key_press(event: mpl.backend_bases.KeyEvent) -> None:
                """
                A function that is called when a key is pressed. If the pressed key is 'x', it closes all the figures.

                :param event: The event object that contains information about the key press.
                :return: None
                """
                if event.key.lower() == 'x':
                    plt.close('all')

            # Make a copy of the weather data
            weather_data: xr.DataArray = self.data[name].copy()
            # Select the desired years and concatenate them along the 'col' dimension
            weather_data: xr.DataArray = xr.concat([weather_data.isel(time=year - 1961) for year in years], dim='col')
            # Select the desired regions
            weather_data: xr.DataArray = weather_data.sel(row=regions)

            z_pre: np.ndarray = weather_data.values
            # Replace comma with dot in the data and convert it to float
            for i in range(z_pre.shape[0]):
                for j in range(z_pre.shape[1]):
                    z_pre[i][j] = z_pre[i][j].replace(',', '.')
            z_pre: np.ndarray = z_pre.astype(float)

            # Create the 2D plot
            fig: mpl.figure.Figure = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)

            # List of rainbow colours
            colors: list[str] = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
            num_boxplots: int = z_pre.shape[0]  # Number of boxplots
            cmap: mpl.colors.LinearSegmentedColormap = mcolors.LinearSegmentedColormap.from_list('rainbow', colors,
                                                                                                 N=num_boxplots)

            box_width: float = 0.5  # Šířka boxplotu

            for i in range(num_boxplots):
                ax.boxplot(z_pre[i, :], patch_artist=True, widths=box_width, positions=[i],
                           boxprops={'facecolor': cmap(i)}, medianprops={'color': 'k'})

            ax.set_xticklabels([self.regions[i] for i in regions], rotation=20, ha='right')
            # Add labels to the axes
            ax.set_xlabel('Kraje [-]', color='blue')
            ax.set_ylabel('Absolutní hodnota spektra ' + ('teploty [°C]' if name == 'temper' else 'srážek [mm]'),
                          color='blue')

            years_str: str = ", ".join(map(str, years[:10])) + (", ..." if len(years) > 10 else "")
            subtitle_part_text: str = "boxploty {} za roky: {}".format(
                "teplot" if name == 'temper' else "srážek", years_str)
            fig.suptitle(subtitle_part_text)
            fig.subplots_adjust(bottom=0.2)

            mpl.rcParams.update(
                {'keymap.forward': ['d', 'D'], 'keymap.back': ['a', 'A'], 'keymap.save': ['s', 'S']})
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            # display the figure
            fig.show()

        def control_years(min_year: int, max_year: int, temp_time_len: int) -> bool:
            """
            Checks if the given range of years is valid or not.

            :param min_year: minimum year in the range
            :param max_year: maximum year in the range
            :param temp_time_len: length of the temperature data
            :return: boolean indicating if the years are valid or not
            """
            if min_year < 1961 or max_year > 1961 + temp_time_len - 1:
                print("Zadané roky nejsou v rozsahu 1961-" + str(1961 + temp_time_len - 1))
                return False
            return True

        def control_regions(regions: list[int]) -> bool:
            """
            Controls whether the regions entered by the user are valid or not.

            :param regions: A list of integers representing regions.
            :return: A boolean indicating whether the regions are valid or not.
            """
            if min(regions) < 0 or max(regions) > 13:
                print("Zadané kraje nejsou v rozsahu 0-13")
                return False
            return True

        # Ask the user to input the years
        years: list[int]
        if years := UserInterface.input_loop("Zadejte rok(y)", year=True):
            # Display the regions
            print("Kraje:")
            print("\n".join([f"{i}) {region}" for i, region in enumerate(self.regions)]))
            # Ask the user to input the regions
            regions: list[int]
            if not (regions := UserInterface.input_loop("Zadejte kraj(e)", region=True)):
                return

            # Validate the input regions
            if not control_regions(regions):
                return

            # Set the minimum and maximum years based on user input
            min_year: int = min(years)
            max_year: int = max(years)

            # Create a 2D boxplots for temperature if selected by user
            if self.temper_choose:
                temp_data: xr.DataArray = self.data['temper'].copy()
                temp_time_len: int = len(temp_data.coords['time'])

                # Validate the input years for temperature
                if not control_years(min_year, max_year, temp_time_len):
                    return
                create_2d_plot("temper", years, regions)

            # Create a 2D boxplots for precipitation if selected by user
            if self.precip_choose:
                prec_data: xr.DataArray = self.data['precip'].copy()
                prec_time_len: int = len(prec_data.coords['time'])

                # Validate the input years for precipitation
                if not control_years(min_year, max_year, prec_time_len):
                    return
                create_2d_plot("precip", years, regions)
            # Show the plot to the user
            plt.show()
        else:
            return

    @Utils.debug
    @reset_data
    def plot_corr_temp_precip_year_region(self) -> None:
        """
        Create and display a correlation plot between temperature and precipitation data for a specific year(s)
        and region(s).

        :return: None
        """

        def create_2d_plot(name: str, years: list = None, regions: list = None):
            """
            Creates a 2D plot that shows the correlation between temperature and precipitation.

            :param name: str, the name of the weather variable ("temper" or "precip")
            :param years: list[int], optional, list of years to plot, if None, all years will be plotted
            :param regions: list[int], optional, list of region IDs to plot, if None, all regions will be plotted
            :return: None
            """
            def on_key_press(event: mpl.backend_bases.KeyEvent) -> None:
                """
                A function that is called when a key is pressed. If the pressed key is 'x', it closes all the figures.

                :param event: The event object that contains information about the key press.
                :return: None
                """
                if event.key.lower() == 'x':
                    plt.close('all')

            def get_weather_data(weather_data: xr.DataArray) -> np.ndarray:
                """
                This function takes a xarray DataArray weather_data and returns a numpy ndarray z_pre
                containing the weather data for the specified years and regions.

                :param weather_data: xarray DataArray containing the weather data
                :return: numpy ndarray containing the weather data for the specified years and regions
                """
                # Select the desired years and concatenate them along the 'col' dimension
                weather_data: xr.DataArray = xr.concat([weather_data.isel(time=year - 1961) for year in years],
                                                       dim='col')
                # Select the desired regions
                weather_data: xr.DataArray = weather_data.sel(row=regions)
                z_pre: np.ndarray = weather_data.values
                # Replace comma with dot in the data and convert it to float
                for i in range(z_pre.shape[0]):
                    for j in range(z_pre.shape[1]):
                        z_pre[i][j] = z_pre[i][j].replace(',', '.')
                return z_pre.astype(float)

            # get weather data
            z_pre1: np.ndarray = get_weather_data(self.data[name].copy())
            name2: str = "temper" if name == "precip" else "precip"
            z_pre2: np.ndarray
            if self.precip_choose and self.temper_choose:
                z_pre2 = get_weather_data(self.data[name2].copy())
            else:
                z_pre2 = z_pre1.copy()

            # Create the 2D plot
            fig: mpl.figure.Figure = plt.figure(figsize=(10, 5))
            years_str: str = ", ".join(map(str, years[:10])) + (", ..." if len(years) > 10 else "")
            temper_or_precip: str
            if self.precip_choose and self.temper_choose:
                temper_or_precip = "srážek na teplotě" if name == "temper" else "teplot na srážkách"
            else:
                temper_or_precip = "srážek na srážkách" if name == "precip" else "teplot na teplotách"
            subtitle_part_text: str = "korelace {} za roky: {}".format(temper_or_precip, years_str)
            fig.suptitle(subtitle_part_text)

            # data normalization
            z_pre1_norm: np.ndarray = (z_pre1 - np.min(z_pre1)) / (np.max(z_pre1) - np.min(z_pre1))
            z_pre2_norm: np.ndarray = (z_pre2 - np.min(z_pre2)) / (np.max(z_pre2) - np.min(z_pre2))

            # Draw a correlation diagram
            plt.scatter(z_pre1_norm, z_pre2_norm)
            xlabel_text: str = "Normované " + ('teploty [-]' if name == "temper" else 'srážky [-]')
            ylabel_text: str
            if self.precip_choose and self.temper_choose:
                ylabel_text = "Normované " + ('srážky [-]' if name == "temper" else 'teploty [-]')
            else:
                ylabel_text = "Normované " + ('teploty [-]' if name == "temper" else 'srážky [-]')
            plt.xlabel(xlabel_text, color='blue')
            plt.ylabel(ylabel_text, color='blue')

            # Perform a linear regression
            slope: Union[np.float64, float]
            intercept: Union[np.float64, float]
            r_value: Union[np.float64, float]
            p_value: Union[np.float64, float]
            std_err: Union[np.float64, float]
            slope, intercept, r_value, p_value, std_err = linregress(z_pre1_norm.ravel(), z_pre2_norm.ravel())

            # Calculate the value of r^2 (coefficient of determination)
            r_squared: Union[np.float64, float] = r_value ** 2

            # Create a regression line
            line_x: np.ndarray = np.array([0, 1])
            line_y: np.ndarray = slope * line_x + intercept

            plt.plot(line_x, line_y, 'r', label='Lineární regrese')
            plt.text(0.025, 0.90, 'Koeficient korelace (r): {:.2f}'.format(r_value), ha='left', va='top',
                     fontsize=10, transform=plt.gca().transAxes, color="#006600", alpha=0.8)
            plt.text(0.025, 0.85, 'Koeficient determinace (r²): {:.2f}'.format(r_squared), ha='left', va='top',
                     fontsize=10, transform=plt.gca().transAxes, color="#006600", alpha=0.8)
            plt.legend()
            plt.grid(True)

            mpl.rcParams.update({'keymap.forward': ['d', 'D'], 'keymap.back': ['a', 'A'], 'keymap.save': ['s', 'S']})
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            # display the figure
            fig.show()

        def control_years(min_year: int, max_year: int, temp_time_len: int) -> bool:
            """
            Checks if the given range of years is valid or not.

            :param min_year: minimum year in the range
            :param max_year: maximum year in the range
            :param temp_time_len: length of the temperature data
            :return: boolean indicating if the years are valid or not
            """
            if min_year < 1961 or max_year > 1961 + temp_time_len - 1:
                print("Zadané roky nejsou v rozsahu 1961-" + str(1961 + temp_time_len - 1))
                return False
            return True

        def control_regions(regions: list[int]) -> bool:
            """
            Controls whether the regions entered by the user are valid or not.

            :param regions: A list of integers representing regions.
            :return: A boolean indicating whether the regions are valid or not.
            """
            if min(regions) < 0 or max(regions) > 13:
                print("Zadané kraje nejsou v rozsahu 0-13")
                return False
            return True

        # Ask the user to input the years
        years: list[int]
        if years := UserInterface.input_loop("Zadejte rok(y)", year=True):
            # Display the regions
            print("Kraje:")
            print("\n".join([f"{i}) {region}" for i, region in enumerate(self.regions)]))
            # Ask the user to input the regions
            regions: list[int]
            if not (regions := UserInterface.input_loop("Zadejte kraj(e)", region=True)):
                return

            # Validate the input regions
            if not control_regions(regions):
                return

            # Set the minimum and maximum years based on user input
            min_year: int = min(years)
            max_year: int = max(years)

            # Create a 2D correlation plot for temperature if selected by user
            if self.temper_choose:
                temp_data: xr.DataArray = self.data['temper'].copy()
                temp_time_len: int = len(temp_data.coords['time'])

                # Validate the input years for temperature
                if not control_years(min_year, max_year, temp_time_len):
                    return
                create_2d_plot("temper", years, regions)

            # Create a 2D correlation plot for precipitation if selected by user
            if self.precip_choose:
                prec_data: xr.DataArray = self.data['precip'].copy()
                prec_time_len: int = len(prec_data.coords['time'])

                # Validate the input years for precipitation
                if not control_years(min_year, max_year, prec_time_len):
                    return
                create_2d_plot("precip", years, regions)
            # Show the plot to the user
            plt.show()
        else:
            return

    @Utils.debug
    @reset_data
    def plot_corr_year_region(self) -> None:
        """
        Create and display a correlation plot between two regions for a specific year(s).

        :return: None
        """
        def create_2d_plot(name: str, years: list = None, region1: int = 0, region2: int = 0):
            """
            Visualize the correlation between two regions for a given weather variable.

            :param name: str, the name of the weather variable ("temper" or "precip")
            :param years: list[int], optional, list of years to plot, if None, all years will be plotted
            :param region1: int, ID of the first region
            :param region2: int, ID of the second region
            :return: None
            """
            def on_key_press(event: mpl.backend_bases.KeyEvent) -> None:
                """
                A function that is called when a key is pressed. If the pressed key is 'x', it closes all the figures.

                :param event: The event object that contains information about the key press.
                :return: None
                """
                if event.key.lower() == 'x':
                    plt.close('all')

            def get_weather_data(weather_data: xr.DataArray, region: int) -> np.ndarray:
                """
                This function takes a xarray DataArray weather_data and returns a numpy ndarray z_pre
                containing the weather data for the specified years and regions.

                :param weather_data: xarray DataArray containing the weather data
                :param region: the ID of the region for which the weather data will be returned
                :return: numpy ndarray containing the weather data for the specified years and regions
                """
                # Select the desired years and concatenate them along the 'col' dimension
                weather_data: xr.DataArray = xr.concat([weather_data.isel(time=year - 1961) for year in years],
                                                       dim='col')
                # Select the desired regions
                weather_data: xr.DataArray = weather_data.sel(row=[region])
                z_pre: np.ndarray = weather_data.values
                # Replace comma with dot in the data and convert it to float
                for i in range(z_pre.shape[0]):
                    for j in range(z_pre.shape[1]):
                        z_pre[i][j] = z_pre[i][j].replace(',', '.')
                return z_pre.astype(float)

            # get weather data
            z_pre1: np.ndarray = get_weather_data(self.data[name].copy(), region1)
            z_pre2: np.ndarray = get_weather_data(self.data[name].copy(), region2)

            # Create the 2D plot
            fig: mpl.figure.Figure = plt.figure(figsize=(10, 5))
            years_str: str = ", ".join(map(str, years[:10])) + (", ..." if len(years) > 10 else "")
            temper_or_precip: str = "teplot" if name == "temper" else "srážek"
            subtitle_part_text: str = "Korelace {} za roky: {}".format(temper_or_precip, years_str)
            fig.suptitle(subtitle_part_text)

            # data normalization
            z_pre1_norm: np.ndarray = (z_pre1 - np.min(z_pre1)) / (np.max(z_pre1) - np.min(z_pre1))
            z_pre2_norm: np.ndarray = (z_pre2 - np.min(z_pre2)) / (np.max(z_pre2) - np.min(z_pre2))

            # Draw a correlation diagram
            plt.scatter(z_pre1_norm, z_pre2_norm)
            plt.xlabel("Normované " + ('teploty' if name == "temper" else 'srážky') + " pro kraj " +
                       self.regions[region1] + " [-]", color='blue')
            plt.ylabel("Normované " + ('teploty' if name == "temper" else 'srážky') + " pro kraj " +
                       self.regions[region2] + " [-]", color='blue')

            # Perform a linear regression
            slope: Union[np.float64, float]
            intercept: Union[np.float64, float]
            r_value: Union[np.float64, float]
            p_value: Union[np.float64, float]
            std_err: Union[np.float64, float]
            slope, intercept, r_value, p_value, std_err = linregress(z_pre1_norm.ravel(), z_pre2_norm.ravel())

            # Calculate the value of r^2 (coefficient of determination)
            r_squared: Union[np.float64, float] = r_value ** 2

            # Create a regression line
            line_x: np.ndarray = np.array([0, 1])
            line_y: np.ndarray = slope * line_x + intercept

            plt.plot(line_x, line_y, 'r', label='Lineární regrese')
            plt.text(0.025, 0.90, 'Koeficient korelace (r): {:.2f}'.format(r_value), ha='left', va='top',
                     fontsize=10, transform=plt.gca().transAxes, color="#006600", alpha=0.8)
            plt.text(0.025, 0.85, 'Koeficient determinace (r²): {:.2f}'.format(r_squared), ha='left', va='top',
                     fontsize=10, transform=plt.gca().transAxes, color="#006600", alpha=0.8)
            plt.legend()
            plt.grid(True)

            mpl.rcParams.update({'keymap.forward': ['d', 'D'], 'keymap.back': ['a', 'A'], 'keymap.save': ['s', 'S']})
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            # display the figure
            fig.show()

        def control_years(min_year: int, max_year: int, temp_time_len: int) -> bool:
            """
            Checks if the given range of years is valid or not.

            :param min_year: minimum year in the range
            :param max_year: maximum year in the range
            :param temp_time_len: length of the temperature data
            :return: boolean indicating if the years are valid or not
            """
            if min_year < 1961 or max_year > 1961 + temp_time_len - 1:
                print("Zadané roky nejsou v rozsahu 1961-" + str(1961 + temp_time_len - 1))
                return False
            return True

        def control_regions(regions: list[int]) -> bool:
            """
            Controls whether the regions entered by the user are valid or not.

            :param regions: A list of integers representing regions.
            :return: A boolean indicating whether the regions are valid or not.
            """
            if min(regions) < 0 or max(regions) > 13:
                print("Zadané kraje nejsou v rozsahu 0-13")
                return False
            return True

        # Ask the user to input the years
        years: list[int]
        if years := UserInterface.input_loop("Zadejte rok(y)", year=True):
            # Display the regions
            print("Kraje:")
            print("\n".join([f"{i}) {region}" for i, region in enumerate(self.regions)]))
            # Ask the user to input the first region
            region1: list[int]
            if not (region1 := UserInterface.input_loop("Zadejte první kraj", region=True)):
                return

            # Validate the input first region
            if not control_regions(region1):
                return
            if len(region1) > 1:
                print("Zadali jste více než jeden kraj")
                return

            # Ask the user to input the second region
            region2: list[int]
            if not (region2 := UserInterface.input_loop("Zadejte druhý kraj", region=True)):
                return

            # Validate the input second region
            if not control_regions(region2):
                return
            if len(region2) > 1:
                print("Zadali jste více než jeden kraj")
                return

            # Set the minimum and maximum years based on user input
            min_year: int = min(years)
            max_year: int = max(years)

            # Create a 2D correlation plot for temperature if selected by user
            if self.temper_choose:
                temp_data: xr.DataArray = self.data['temper'].copy()
                temp_time_len: int = len(temp_data.coords['time'])

                # Validate the input years for temperature
                if not control_years(min_year, max_year, temp_time_len):
                    return
                create_2d_plot("temper", years, region1.copy()[0], region2.copy()[0])

            # Create a 2D correlation plot for temperature if selected by user
            if self.precip_choose:
                prec_data: xr.DataArray = self.data['precip'].copy()
                prec_time_len: int = len(prec_data.coords['time'])

                # Validate the input years for precipitation
                if not control_years(min_year, max_year, prec_time_len):
                    return
                create_2d_plot("precip", years, region1.copy()[0], region2.copy()[0])
            # Show the plot to the user
            plt.show()
        else:
            return


class DataPlotter:
    """
    The DataPlotter class is responsible for plotting various graphs based on the temperature and
    precipitation data provided.
    The class has a switch_case() method that calls different plot-generating methods based on
    the user's choice of plot type.
    Overall, the DataPlotter class provides a flexible and intuitive interface for visualizing climate data.
    """

    def __init__(self):
        # initialize instance variables
        self.data: dict = {}
        self.temper_data_are: bool = False
        self.precip_data_are: bool = False
        self.temper_choose: bool = False
        self.precip_choose: bool = False

    @staticmethod
    def gen_index() -> Generator[int, None, None]:  # index generator
        """
        Generator function for generating indices

        :return: returns a generator object which generates integers from 0 to 9
        """
        for i in range(10):
            yield i

    def temper_or_precip(self) -> str:
        """
        This method checks which type of data the user has chosen to plot - temperature and/or precipitation.
        It sets a string variable depending on the user's choice.

        :return: returns the final string
        """
        output_string: str = ""
        if self.temper_choose:
            output_string = "teplot"
            if self.precip_choose:
                output_string += " a srážek"
        elif self.precip_choose:
            output_string = "srážek"
        return output_string

    def switch_case(self, key: str) -> None:
        """
        Executes different plot generating methods based on user input.

        :param key: user's choice of plot type as string number
        :return: None
        """
        graph_plotter = GraphPlotter(self.data, self.temper_choose, self.precip_choose)
        cases: dict = {
            "0": graph_plotter.plot_3d,
            "1": graph_plotter.plot_3d_year,
            "2": graph_plotter.plot_3d_year_region,
            "3": graph_plotter.plot_2d_year_region,
            "4": graph_plotter.plot_3d_hist_dct,
            "5": graph_plotter.plot_3d_hist_dct_year_region,
            "6": graph_plotter.plot_2d_hist_dct_year_region,
            "7": graph_plotter.plot_boxplot_year_region,
            "8": graph_plotter.plot_corr_temp_precip_year_region,
            "9": graph_plotter.plot_corr_year_region,
            "default": lambda: print("Vybrali jste zpět"),
        }
        cases.get(key, cases["default"])()

    def plot_data(self, data: dict) -> None:
        # Make a copy of the input dictionary and store it as an instance attribute
        self.data: dict = data.copy()

        # Check if temperature and precipitation data are in the input dictionary
        if "temper" in data:
            self.temper_data_are = True
            print("Byla načtena data teplot")
        if "precip" in data:
            self.precip_data_are = True
            print("Byla načtena data srážek")
        print()

        # Loop until the user chooses to stop
        while True:
            # Ask the user if they want to work with temperature data
            if self.temper_data_are:
                self.temper_choose = True if DEBUG_SKIP else UserInterface.input_loop("Chcete pracovat s teplotami?")

            # Ask the user if they want to work with precipitation data
            if self.precip_data_are:
                self.precip_choose = True if DEBUG_SKIP else UserInterface.input_loop("Chcete pracovat se srážkami?")

            # If the user doesn't choose to work with either data, ask again
            if not (self.temper_choose or self.precip_choose):
                print("Jiné data aktuálně nezpracovávám, rozhodněte se ještě jednou\n")
                continue

            # Loop until the user chooses to go back
            while True:
                # Generate a list of options based on the user's choices
                idx: Generator[int, None, None] = self.gen_index()
                print(f"""\nMožnosti:
{next(idx)}) 3D graf {self.temper_or_precip()} za posledních 5 let
{next(idx)}) 3D graf {self.temper_or_precip()} pro zvolené rok(y)
{next(idx)}) 3D graf {self.temper_or_precip()} pro zvolené rok(y) a region(y)
{next(idx)}) 2D graf {self.temper_or_precip()} pro zvolené rok(y) a region(y) 
{next(idx)}) 3D histogram DCT {self.temper_or_precip()} za posledních 5 let
{next(idx)}) 3D histogram DCT {self.temper_or_precip()} pro zvolené rok(y) a region(y)
{next(idx)}) 2D histogram DCT {self.temper_or_precip()} pro zvolené rok(y) a region(y)
{next(idx)}) Boxploty {self.temper_or_precip()} pro zvolené rok(y) a region(y)
{next(idx)}) Korelační graf {self.temper_or_precip()} na sobě pro zvolené rok(y) a region(y)
{next(idx)}) Korelační graf {self.temper_or_precip()} pro zvolené rok(y) mezi 2 regiony
jiné číslo) zpět
""")
                # Get the user's choice and call the appropriate graph using switch_case()
                match: str = UserInterface.input_loop("Vyberte jednu z možností", match=True)
                self.switch_case(match)

                print()
                # Ask the user if they want to continue
                if not UserInterface.input_loop("Chcete pokračovat ve vykreslování?"):
                    break


if __name__ == "__main__":
    # Sets up a keyboard interrupt signal handler.
    signal.signal(signal.SIGINT, Utils.handle_keyboard_interrupt)

    # Prints out a welcome message.
    Utils.welcome()

    # Creates instances of the DataFetcher and DataPlotter classes.
    fetcher = DataFetcher()
    plotter = DataPlotter()

    # Loops until the user exits the program.
    while True:
        # Asks the user if they are online and if they want temperature and precipitation data.
        online: bool = UserInterface.input_loop("Jste online") if not DEBUG_SKIP else False
        temperatures: bool = UserInterface.input_loop("Chcete data teplot") if not DEBUG_SKIP else True
        precipitation: bool = UserInterface.input_loop("Chcete data srážek") if not DEBUG_SKIP else True

        # If the user does not want any data, prompts them to try again or exit the program.
        if not (temperatures or precipitation):
            print("Jiné data aktuálně nezpracovávám, rozhodněte se ještě jednou nebo "
                  "pro ukončení použijte 'x'.\n")
            continue

        # Asks the user if they want to fetch data in parallel, if they are online.
        parallel: bool = False
        if online:  # only in online mode
            parallel: bool = UserInterface.input_loop("Chcete data načítat paralelně")
        print()

        # Fetches the data from the API and stores it in a dictionary.
        try:
            data: dict = fetcher.get_data(online, temperatures, precipitation, parallel)
        except JumpException:
            print()
            continue

        # If the data was fetched successfully, plots it using the DataPlotter.
        if data:
            plotter.plot_data(data)
        print()
