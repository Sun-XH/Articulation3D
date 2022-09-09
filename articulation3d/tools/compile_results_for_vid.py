import os, glob
import pdb
import shutil

import dominate
from dominate.tags import *

from collections import defaultdict
import argparse

def parse_vid_and_cat(vid):
    cat = '_'.join(vid.split('/')[:-1])
    vid_num = vid.split('/')[-1]

    return cat, vid_num

def main(args):
    videos = sorted(glob.glob(os.path.join(args.in_path, '*/*')))
    videos_path = []
    for vid_tmp in videos:
        vid = vid_tmp[18:]
        videos_path.append(vid)

    doc = dominate.document(title=f'3DADN - Results')

    cat_vids = defaultdict(list)
    # pdb.set_trace()
    for vid in videos_path:

        cat, vid_num = parse_vid_and_cat(vid)
        cat_vids[cat].append(vid_num)

    # pdb.set_trace()
    for k, v in cat_vids.items():
        doc += a(f'{k} ({len(v)})', href=f'#{k}')
        doc += br()

    for k, v in cat_vids.items():

        cat_link = a(id=f'{k}')
        cat_link.add(h2(f'{k} ({len(v)})'))
        doc.add(cat_link)

        
        for vid in v:
            vid_link = a(f'{k}/{vid}' ,href=f"{os.path.join(k, vid, 'final_result.gif')}")
            vid_link.add(br())
            vid_link.add(img(src=f"{os.path.join(k, vid, 'final_result.gif')}"))
            doc.add(vid_link)
            doc.add(br())
            doc.add(hr())

            os.makedirs(os.path.join(args.out_path, k, vid), exist_ok=True)
            # pdb.set_trace()
            shutil.copy(os.path.join(args.in_path, f'{k}/{vid}', args.in_folder, 'final_result.gif'),
                        os.path.join(args.out_path, k, vid, 'final_result.gif'))
    print(doc.render())
    with open(os.path.join(args.out_path, 'viz.html'), 'w') as fp:
        fp.write(doc.render())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--in_folder', type=str, default='')
    # parser.add_argument('--desc', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    main(args)