import zipfile
import json
import struct
import io
import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

dtype2struct = {'uint8': 'B', 'uint32': 'I', 'float64': 'd', 'float32': 'f'}
dtype2length = {'uint8': 1, 'uint32': 4, 'float64': 8, 'float32': 4}


def import_smlm(file_path):
    zf = zipfile.ZipFile(file_path, 'r')
    file_names = zf.namelist()
    if "manifest.json" in file_names:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest['format_version'] == '0.2'
        for file_info in manifest['files']:
            if file_info['type'] == "table":
                logger.info('loading table...')
                format_key = file_info['format']
                file_format = manifest['formats'][format_key]
                if file_format['mode'] == 'binary':
                    try:
                        table_file = zf.read(file_info['name'])
                        logger.info(file_info['name'])
                    except KeyError:
                        logger.error('ERROR: Did not find %s in zip file', file_info['name'])
                        continue
                    else:
                        logger.info('loading table file: %s bytes', len(table_file))
                        logger.info('file format: %s', file_format)
                        headers = file_format['headers']
                        dtype = file_format['dtype']
                        shape = file_format['shape']
                        hLen = len(headers)
                        assert len(headers) == len(dtype) == len(shape)
                        rowLen = 0
                        for i, h in enumerate(file_format['headers']):
                            rowLen += dtype2length[dtype[i]]
                        rows = file_info['rows']
                        tableDict = {}
                        byteOffset = 0
                        try:
                            import numpy as np
                            for i, h in enumerate(file_format['headers']):
                                tableDict[h] = np.ndarray((rows,), buffer=table_file, dtype=dtype[i], offset=byteOffset, order='C', strides=(rowLen,))
                                byteOffset += dtype2length[dtype[i]]
                        except ImportError:
                            logger.warning('Failed to import numpy, performance will drop dramatically. Please install numpy for the best performance.')
                            st = ''
                            for i, h in enumerate(file_format['headers']):
                                st += (str(shape[i])+dtype2struct[dtype[i]])

                            unpack = struct.Struct(st).unpack
                            tableDict = {h:[] for h in headers}
                            for i in range(0, len(table_file), rowLen):
                                unpacked_data = unpack(table_file[i:i+rowLen])
                                for j, h in enumerate(headers):
                                    tableDict[h].append(unpacked_data[j])
                            tableDict = {h:np.array(tableDict[h]) for i,h in enumerate(headers)}
                        data = {}
                        data['min'] = [tableDict[h].min() for h in headers]
                        data['max'] = [tableDict[h].max() for h in headers]
                        data['avg'] = [tableDict[h].mean() for h in headers]
                        data['tableDict'] = tableDict
                        file_info['data'] = data
                        logger.info('table file loaded: %s', file_info['name'])
                else:
                    raise Exception('format mode {} not supported yet'.format(file_format['mode']))
            elif file_info['type'] == "image":
                if file_format['mode'] == 'binary':
                    try:
                        image_file = zf.read(file_info['name'])
                        logger.info('image file loaded: %s', file_info['name'])
                    except KeyError:
                        logger.error('ERROR: Did not find %s in zip file', file_info['name'])
                        continue
                    else:
                        from PIL import Image
                        image = Image.open(io.BytesIO(image_file))
                        data = {}
                        data['image'] = image
                        file_info['data'] = data
                        logger.info('image file loaded: %s', file_info['name'])

            else:
                logger.info('ignore file with type: %s', file_info['type'])
    else:
        raise Exception('invalid file: no manifest.json found in the smlm file')
    return manifest, manifest['files']


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from plot_hist import plotHist
    import time
    start_time = time.time()
    # Download a file from: https://shareloc.xyz/#/repository by clicking the `...` button of the file and click the name contains `.smlm`
    manifest, files = import_smlm('./localization_table.smlm')
    logger.info("--- file loaded in %s seconds ---" % (time.time() - start_time))
    h = plotHist(files[0]['data']['tableDict'], value_range=(0,10))
    plt.figure(figsize=(20,20))
    plt.imshow(h)
    plt.savefig('./localization_histogram.png')
    # plt.figure(figsize=(20,20))
    # plt.imshow(np.array(files[1]['data']['image']))
    logger.info("--- file displayed in %s seconds ---" % (time.time() - start_time))
    # Experiment: 7GB csv file --> 21MB smlm file; loading time: 21s with python-numpy implemetation, 116s with pure python implementation.
