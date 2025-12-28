from revu.oa_downloader.download_2 import OaDownloader
from pathlib import Path

test_cache_dir = str(Path.cwd() / 'tests/data/oa_downloader')
test_downloader = OaDownloader(test_cache_dir, ['cc-by', 'cc0', 'public-domain', 'cc-by-sa'])

def test_download_one():
    result = test_downloader.download_oa_pdf(0, 'https://www.mdpi.com/1660-4601/18/21/11447/pdf?version=1635759585', 1)
    print(result)
    is_not_equal(None, result)

def is_equal(expected, actual):
    assert expected == actual, f"Expected: {expected}, Actual: {actual}"
    print(f"✓ Assertion passed")

def is_not_equal(expected, actual):
    assert expected != actual, f"Expected: {expected}, Actual: {actual}"
    print(f"✓ Assertion passed")

if __name__ == '__main__':
    test_download_one()
