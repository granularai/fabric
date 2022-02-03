import random
import numpy as np
import rasterio as rio
import os, shutil, glob, json
import cv2

from PIL import Image
from torch.utils.data import Dataset

from shapely.geometry import Polygon
from skimage.draw import polygon

from phobos.grain import Grain
from phobos.io import getDataLoaders
from phobos.transforms import Normalize

def get_image_meta(dsetdir):
    cogpath = os.path.join(dsetdir,'tiles')
    tifpath = os.path.join(dsetdir,'images') 
    annpath = os.path.join(dsetdir,'annotations') 

    meta = {}

    for apath in glob.glob(f'{annpath}/*.json'):
        key = apath.split('/')[-1].split('.')[0]
        
        tpath = os.path.join(tifpath,key)
        cpath = os.path.join(cogpath,key)
        
        meta[key] = {}

        meta[key]['ann'] = apath
        meta[key]['cog'] = cpath
        meta[key]['tif'] = tpath

    return meta

def get_full_load(meta):
    dates =  {
        1: 'imgs_1',
        2: 'imgs_mid_1',
        3: 'imgs_mid_2',
        4: 'imgs_mid_1',
        5: 'imgs_2'
    }

    load = {}

    for key in meta:
        load[key] = {}

        afp  = open(meta[key]['ann'],'r')
        amap = json.load(afp)

        jpath = glob.glob(f"{meta[key]['cog']}/*.jp2")[0]
        jrstr = rio.open(jpath)
        
        jparr = jrstr.read()
        jtrns = jrstr.transform 

        h,w  = jparr.shape[1:]
        mask = np.zeros((h,w),dtype=np.uint8)

        for annotation in amap['annotations']:
            poly = Polygon(annotation['geometry']['coordinates'][0])
            
            xs,ys = poly.exterior.coords.xy
            rc    = rio.transform.rowcol(jtrns,xs,ys)
            poly  = np.asarray(list(zip(rc[0],rc[1])))
            rr,cc = polygon(poly[:,0],poly[:,1],mask.shape)
            mask[rr,cc] = 1

        mask = mask.transpose().astype(np.uint8)

        imap = {}
        for dkey in dates:
            dpath = os.path.join(meta[key]['tif'],dates[dkey])

            imgnpath = glob.glob(f'{dpath}/*B08.tif')[0]
            imgrpath = glob.glob(f'{dpath}/*B04.tif')[0]
            imggpath = glob.glob(f'{dpath}/*B03.tif')[0]
            imgbpath = glob.glob(f'{dpath}/*B02.tif')[0]

            rrstr = rio.open(imgrpath)
            grstr = rio.open(imggpath)
            brstr = rio.open(imgbpath)
            nrstr = rio.open(imgnpath)

            rarr = cv2.resize(rrstr.read()[0].astype(np.float32),(h,w))
            garr = cv2.resize(grstr.read()[0].astype(np.float32),(h,w))
            barr = cv2.resize(brstr.read()[0].astype(np.float32),(h,w))
            narr = cv2.resize(nrstr.read()[0].astype(np.float32),(h,w))

            tile = np.stack([rarr,garr,barr,narr],axis=0)
            
            imap[dkey] = tile
        
        load[key] = {'imgs': imap, 'mask': mask}

    return load

def get_train_val_keys(meta, full_load, shape, args):
    p = shape
    s = args.stride
    r = args.ratio
    
    th = args.thres

    keys = list(meta.keys())
    random.shuffle(keys)

    tkeys = keys[:int(r*len(keys))]
    vkeys = keys[int(r*len(keys)):]

    tkeylist,vkeylist = [],[]

    for key in keys:
        mask = full_load[key]['mask']
        
        h,w = mask.shape
        keylist = [[key,i,j] 
                    for i in range(0,h,s) \
                        for j in range(0,w,s) \
                            if i+p<h and j+p<w \
                                and np.sum(mask[i:i+p,j:j+p]) > th]
        
        if key in tkeys:
            tkeylist.extend(keylist)
        elif key in vkeys:
            vkeylist.extend(keylist)

    return tkeylist,vkeylist

def get_sample(full_load,key,x,y,shape,args):
    s = shape
    load = full_load[key]
    ibands = args.input.heads['rgbn']['bands']

    means = [ibands[band]['mean'] for band in ibands]
    stds  = [ibands[band]['std'] for band in ibands]

    N = Normalize(mean=means,std=stds)

    ipatchlist = []
    for date in load['imgs']:
        tile  = load['imgs'][date]
        patch = np.copy(tile[:,x:x+s,y:y+s]).transpose((1,2,0))

        patch = N.apply(patch).transpose((2,0,1))

        ipatchlist.append(patch)

    ipatch = np.stack(ipatchlist,axis=0)

    mask   = load['mask']
    mpatch = mask[x:x+s,y:y+s]

    inputs = { 'rgbn': ipatch }
    labels = { 'cd': mpatch }

    return inputs, labels

class OSCDDataset(Dataset):
    def __init__(self,samples,full_load,args,shape):
        random.shuffle(samples)

        self.args = args
        self.shape = shape
        self.loader = get_sample
        self.samples = samples
        self.full_load = full_load

        
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) :
        key,x,y = self.samples[index]

        return get_sample(
                        full_load=self.full_load,
                        key=key,x=x,y=y,
                        shape=self.shape,
                        args=self.args
                )

def getLoadersMap(args,inputs):
    shape = inputs.heads['rgbn'].shape.H

    oscd_meta = get_image_meta(args.dataset_path)
    oscd_full_load = get_full_load(oscd_meta)

    tkeys, vkeys = get_train_val_keys(oscd_meta,oscd_full_load,shape,args)

    print(f'number of train keys : {len(tkeys)}')
    print(f'number of val keys : {len(vkeys)}\n')

    datasets = {
        'train': OSCDDataset(
                        samples=tkeys,
                        full_load=oscd_full_load,
                        args=args,shape=shape
                ),
        'val': OSCDDataset(
                        samples=vkeys,
                        full_load=oscd_full_load,
                        args=args,shape=shape
            )
    }

    loaders = getDataLoaders(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=args.distributed,
        load=args.load
    )

    return loaders
