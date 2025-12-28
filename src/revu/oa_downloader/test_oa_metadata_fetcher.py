from revu.oa_downloader.oa_metadata_fetcher import OaMetadataFetcher
from pathlib import Path

test_cache_dir = str(Path.cwd() / 'tests/data/oa_downloader')
test_fetcher = OaMetadataFetcher('gaugabgo.dev@gmail.com', test_cache_dir)

def test_integration():
    
    test_dois = [
        '10.1016/j.envres.2020.109292',
        '10.1016/j.cardfail.2021.11.023',
        '10.1016/j.envint.2017.12.033'
    ]

    results = test_fetcher.fetch_all_metadata(test_dois)
    is_equal(3, len(results))
    test_fetcher.build_metdata_table()

def test_fetch_one_metadata():
    result = test_fetcher.fetch_one_metadata('10.1016/j.envres.2020.109292')
    is_not_equal(None, result)
    is_equal(200, result['status_code'])
    is_not_equal(None, result['metadata'])
    is_not_equal(None, result['metadata']['doi'])

def test_fetch_one_metadata_invalid():
    result = test_fetcher.fetch_one_metadata('trash')
    is_not_equal(None, result)
    is_equal(404, result['status_code'])
    is_equal(None, result['metadata'])

def is_equal(expected, actual):
    assert expected == actual, f"Expected: {expected}, Actual: {actual}"
    print(f"✓ Assertion passed")

def is_not_equal(expected, actual):
    assert expected != actual, f"Expected: {expected}, Actual: {actual}"
    print(f"✓ Assertion passed")

if __name__ == '__main__':
    test_fetch_one_metadata()
    test_fetch_one_metadata_invalid()
    test_integration()
