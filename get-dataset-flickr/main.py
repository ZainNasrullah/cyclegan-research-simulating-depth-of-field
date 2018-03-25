from flickrDownloader import *
from flickrDownloader.flickrDownloader import *
from flickrDownloader.utils import *


# if you want to insert your apikey in source code
api_key = "fbc2844adebc28a5ce4ab7af3bc1de1b"
# api_key = "flickr.apikey"  # if you want to read apikey from file

# If you want to share your code in git, you may want not to share your api key too!
# In that case, insert your api key in the flickr.apikey file and add flickr.apikey in your .gitignore


# Available licenses: (from: https://www.flickr.com/services/api/explore/flickr.photos.licenses.getInfo)
#
# {"id": 0, "name": "All Rights Reserved", "url": ""},
# {"id": 4, "name": "Attribution License", "url": "https:\/\/creativecommons.org\/licenses\/by\/2.0\/"},
# {"id": 6, "name": "Attribution-NoDerivs License", "url": "https:\/\/creativecommons.org\/licenses\/by-nd\/2.0\/"},
# {"id": 3, "name": "Attribution-NonCommercial-NoDerivs License", "url": "https:\/\/creativecommons.org\/licenses\/by-nc-nd\/2.0\/"},
# {"id": 2, "name": "Attribution-NonCommercial License", "url": "https:\/\/creativecommons.org\/licenses\/by-nc\/2.0\/"},
# {"id": 1, "name": "Attribution-NonCommercial-ShareAlike License",  "url": "https:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/"},
# {"id": 5, "name": "Attribution-ShareAlike License", "url": "https:\/\/creativecommons.org\/licenses\/by-sa\/2.0\/"},
# {"id": 7, "name": "No known copyright restrictions", "url": "https:\/\/www.flickr.com\/commons\/usage\/"},
# {"id": 8, "name": "United States Government Work", "url": "http:\/\/www.usa.gov\/copyright.shtml"},
# {"id": 9, "name": "Public Domain Dedication (CC0)", "url": "https:\/\/creativecommons.org\/publicdomain\/zero\/1.0\/"},
# {"id": 10, "name": "Public Domain Mark", "url": "https:\/\/creativecommons.org\/publicdomain\/mark\/1.0\/"}
license_id = 10  # "using public domain mark" license

query = "portrait bokeh -blackandwhite -BW -monochrome -animal -cat -dog -bird -flower"

query2 = r"portrait"

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
