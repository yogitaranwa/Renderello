# Per-source capture-origin verification (owning-page evidence, 2026-11-20 ~19:38 IST)
Method: fetched each source's owning page. Flickr /meta/ pages show the embedded EXIF (camera model, taken-on date, software); WordPress Photo Directory pages show EXIF blocks; rawpixel pages show the site's media-type label; Wikimedia shows the file metadata disclaimer.

## Camera-EXIF verified real photographs (13)
| benchmark label | owning page | camera / evidence |
|---|---|---|
| backpack2 | flickr.com/photos/24226200@N00/1466198389/meta/ | Canon EOS Digital Rebel XTi, taken 2007-09-30 |
| can1 | flickr.com/photos/55608722@N06/12199880715/meta/ | Nikon D3S, 50mm f/1.8, taken 2014-01-28 |
| can2 | flickr.com/photos/33907867@N02/4306184928/meta/ | Canon PowerShot G10 via Ortery Photosimile (automated photo studio), taken 2010-01-26 |
| glass1 | flickr.com/photos/37387065@N05/5080257300/meta/ | Canon EOS REBEL T2i, EF50mm, taken 2010-10-13 |
| perfume2 | flickr.com/photos/22141297@N06/7796400432/meta/ | Nikon Coolpix S4000, taken 2012-08-16 |
| watch1 | flickr.com/photos/49503023809@N01/2418089346/meta/ | Nikon D80, taken 2008-04-15 |
| watch2 | flickr.com/photos/51764518@N02/15757858992/meta/ | Polaroid iS2132, taken 2014-11-09 |
| wine1 | flickr.com/photos/13203757@N00/2879471624/meta/ | Nikon D80, taken 2008-09-22 |
| wine2 | flickr.com/photos/81266640@N00/3885797185/meta/ | Nikon D300, taken 2009-09-04 |
| lotion1 | flickr.com/photos/53719740@N05/8706603913/meta/ | Canon EOS-1Ds Mark III, taken 2013-03-18 |
| lotion2 | flickr.com/photos/85171915@N04/8056127582/meta/ | Canon EOS 1000D, taken 2012-07-10 |
| plush_boba1 | wordpress.org/photos/photo/25269fcc28/ | EXIF f/1.7, 6.3mm, ISO 125, 1/50s; caption "Captured indoors in Kurla, Mumbai" |
| multiproduct_barber1 | wordpress.org/photos/photo/27966437d7/ | EXIF f/7.1, 18mm, ISO 1600, 1/60s, 6000x4000 |

## Photographic, weaker page-level evidence (8)
| benchmark label | owning page | evidence |
|---|---|---|
| backpack1 | flickr.com/photos/92167476@N03/9418654172/meta/ | EXIF block present, taken 2013-08-01; no camera model recorded |
| glass2 | flickr.com/photos/60291025@N05/10841624814/meta/ | IPTC envelope + XMP (Photoshop CS6), taken 2013-11-13; no camera model |
| shampoo2 | commons.wikimedia.org/wiki/File:Shampoo_Bottle_made_of_PLA-Blend_Bio-Flex.jpg | Wikimedia reports embedded EXIF metadata exists; FKuR company product photo, VRT-verified 2009 |
| facemask | rawpixel.com/image/11524180/... | rawpixel labels media type "Photo", JPEG 4460x2973 @300dpi, public-domain photo collection |
| sneaker1 | rawpixel.com/image/11524416/... | same site-level "Photo" label |
| sneaker2 | rawpixel.com/image/11524257/... | same site-level "Photo" label |
| skincare_tinytext1 (manifest label "amber dropper bottle") | rawpixel.com/image/11524368/back-side-beard-oil-bottle | same site-level "Photo" label |
| furry_teddy1 | rawpixel.com/image/6017419/... | same site-level "Photo" label, JPEG 5540x3918 @300dpi |

## NOT real photographs (3)
| benchmark label | owning page | evidence |
|---|---|---|
| snack1 | flickr.com/photos/42042252@N02/5559242063/meta/ | EXIF Software: CINEMA 4D -> CGI render (matches benchmark classification) |
| tshirt1 | flickr.com/photos/203051018@N08/54591299167/meta/ | page title "Tshirt Mockup", no camera EXIF -> mockup |
| tshirt2 | flickr.com/photos/203051018@N08/54592371899/meta/ | page title "Tshirt Mockup", no camera EXIF -> mockup |

## Tally
24 sources: 21 photographs (13 camera-EXIF-verified, 8 site-level photographic evidence), 2 t-shirt mockups, 1 CGI render. Benchmark classifications for snack1/tshirt1/tshirt2 confirmed by owning-page evidence. The "20+ real photos" coverage claim is supported at 21/24, with 13 of those camera-verified.
