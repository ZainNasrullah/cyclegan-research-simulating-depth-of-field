from flickrDownloader import *
from flickrDownloader.flickrDownloader import *
from flickrDownloader.utils import *


# if you want to insert your apikey in source code
api_key = ""
license_id = 10  # "using public domain mark" license

query = "portrait bokeh -blackandwhite -BW -monochrome -animal -cat -dog -bird -flower"
query2 = r"portrait"

flickr_photos_downloader(api_key,
                         n_images=4000,
                         query_text=query,
                         tag_mode=FlickrTagMode.any,
                         image_size=FlickrImageSize.longedge_1600,
                         content_type=FlickrContentType.photos,
                         media=FlickrMedia.photos,
                         download_path="bokeh_downloads",
                         save_filename_prefix="bokeh_downloaded_",
                         forced_extension=None,
                         verbose=True,
                         ignore_errors=False,
                         license_id=license_id)

flickr_photos_downloader(api_key,
                         n_images=4000,
                         query_text=query2,
                         tag_mode=FlickrTagMode.any,
                         image_size=FlickrImageSize.longedge_1600,
                         content_type=FlickrContentType.photos,
                         media=FlickrMedia.photos,
                         download_path="iphone_downloads",
                         save_filename_prefix="iphone_downloaded_",
                         forced_extension=None,
                         verbose=True,
                         ignore_errors=False,
                         license_id=license_id,
                         camera=r'&camera=apple%2Fiphone_7')
