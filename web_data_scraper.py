from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
import urllib.request
import os.path

WANTED_COUNTRIES = ['england', 'germany', 'italy', 'spain', 'france', 'portugal', 'netherlands']
SEASON_TO_DOWNLOAD_COUNT = 10


def download_data_from_web(download_to_path):
    web_handler: WebDriver = webdriver.Chrome()
    try:
        web_handler.get("http://www.football-data.co.uk/data.php")
        league_link_elements = web_handler.find_elements_by_xpath("//td[@valign='top'][2]//table//a")
        league_links = []
        for element in league_link_elements:
            league_links.append(element.get_attribute("href"))
        for link in league_links:
            if any((country in link) for country in WANTED_COUNTRIES):
                download_info_about_matches(web_handler, link, download_to_path)
    finally:
        web_handler.close()


def download_info_about_matches(web_handler, link, download_to_path):
    web_handler.get(link)
    season_headers = web_handler.find_elements_by_xpath("//td/i")
    for i, header in enumerate(season_headers):
        if i < SEASON_TO_DOWNLOAD_COUNT:
            season_years = header.text.split()[1].replace("/", "")
            league_in_curr_season_download_elements = header.find_elements_by_xpath("./following-sibling::a[position()<=2]")
            for league_download_element in league_in_curr_season_download_elements:
                download_link = league_download_element.get_attribute("href")
                extension = download_link.split(".")[-1]
                full_file_name = download_to_path + "\\" + league_download_element.text + season_years + "." + extension
                if i == 0 or not(os.path.isfile(full_file_name)):
                    urllib.request.urlretrieve(download_link, full_file_name)
