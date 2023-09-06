nhrs = 240

attrs <- list(
        summary = 'Surface influence fields (footprints) stored in sparse format (only the non-zero values)',
        model = 'STILT (v1.2), source code provided by Thomas Koch from MPI BGC',
        institution = 'Wageningen University, department of Meteorology and Air Quality, Wageningen, the Netherlands; Rijksuniversiteit Groningen, Groningen, the Netherlands; ICOS Carbon Portal, Lund, Sweden',
        contact = 'Daan Kivits; daan.kivits@wur.nl',
        conventions = 'CF-1.8',
        creation_date = format(Sys.time(), "%Y-%m-%d %H:%M"),
        crs = 'spherical earth with radius of 6370 km',
        disclaimer = 'This data belongs to the CarbonTracker project',
        history = paste('File created on', format(Sys.time(), "%Y-%m-%d %H:%M"),
                        'by dkivits, using the code on  the Subversion (SVN) repository on https://projects.bgc-jena.mpg.de/STILT/svn.
                        R version', R.version$version.string),
        creator = 'Daan Kivits, https://orcid.org/0009-0005-8856-8497',
        frequency = '1h',
        length = paste(str(nhrs), 'h'),
        grid_definition = 'Sparse format of CTE-HR grid, which ranges from -15E to 35W and 33N to 72N, with a resolution of 0.1x0.2 degree',
        geospatial_lat_resolution = '0.1 degree',
        geospatial_lon_resolution = '0.2 degree',
        keywords = 'footprint, surface influence field, STILT',
        license = 'CC-BY-4.0',
        nominal_resolution = '0.1x0.2 degree'
)

for (attr in names(attrs)){
    print(attr)
    print(attrs[[attr]])
}