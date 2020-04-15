import os

import json
from basecamp.grain.grain import Grain



def get_args(experiment=None, metadata_json='metadata.json'):
    """
    Initialize Parser and define arguments
    """

    grain_exp = Grain(polyaxon_exp=experiment)

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        grain_exp.set_defaults(**metadata)

        args = grain_exp.parse_args(['--sensor', 'sentinel2',
                                      '--band_ids', 'B02,B03,B04,B08',
                                      '--input_shape', '1,48,48,4',
                                      '--resolution', '1'])
        return args

    return None
