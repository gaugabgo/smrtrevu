from revu.oa_downloader.download_2 import OaDownloader

def test_download_integration():
    cache_dir = 'tests/data/oa_downloader/cache'
    allowed_licenses = ['cc-by', 'cc0', 'public-domain', 'cc-by-sa']
    input = 'tests/data/oa_downloader/metadata_test.csv'

    downloader = OaDownloader(cache_dir, allowed_licenses, download_method='requests')
    downloader.run(input)

def unit_downloader():
    cache_dir = 'tests/data/oa_downloader/cache'
    allowed_licenses = ['cc-by', 'cc0', 'public-domain', 'cc-by-sa']
    pdf_link = "http://www.scielo.br/pdf/csc/v24n7/1413-8123-csc-24-07-2649.pdf"
    filename = "10.1590_1413-81232018247.15662017.pdf"
    downloader = OaDownloader(cache_dir, allowed_licenses, download_method='curl')
    downloader.download_oa_pdf(0, pdf_link, filename)

def main():
    test_download_integration()

if __name__ == '__main__':
    main()