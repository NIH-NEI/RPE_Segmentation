import  sys, os

import imagetools

from rpe_config import RPE_Config
from rpe_stack import RpeStack
from predict import iter_multi_stacks, RPE_Predictor

def to_text(title, base_cnt, cmp_cnt, f_pos, f_neg, fragm, fused, pct_match):
    n_match = int(sum(pct_match))
    n_total = int(f_neg + fragm + fused + n_match)
    if n_total <= 0: n_total = 1
    pct_pri = base_cnt * 100. / n_total
    pct_sec = cmp_cnt * 100. / n_total
    pct_f_pos = f_pos * 100. / n_total
    pct_f_neg = f_neg * 100. / n_total
    pct_fragm = fragm * 100. / n_total
    pct_fused = fused * 100. / n_total
    lines = []
    lines.append('+-------------------------------+---------+----------+')
    lines.append(f'| {title:50s} |')
    lines.append('+-------------------------------+---------+----------+')
    lines.append(f'| Primary annotations           | {base_cnt:7d} | {pct_pri:7.2f}% |')
    lines.append(f'| Secondary annotations         | {cmp_cnt:7d} | {pct_sec:7.2f}% |')
    lines.append(f'| Fragmented                    | {fragm:7d} | {pct_fragm:7.2f}% |')
    lines.append(f'| Fused                         | {fused:7d} | {pct_fused:7.2f}% |')
    lines.append(f'| False Positives               | {f_pos:7d} | {pct_f_pos:7.2f}% |')
    lines.append(f'| False Negatives               | {f_neg:7d} | {pct_f_neg:7.2f}% |')
    lines.append('+-------------------------------+---------+----------+')
    for iou in (95, 90, 80, 75, 50):
        iou_cnt = int(sum(pct_match[iou:]))
        iou_pct = iou_cnt * 100. / n_total
        lines.append(f'| Matches at IoU >={iou:3d}%         | {iou_cnt:7d} | {iou_pct:7.2f}% |')
    lines.append('+-------------------------------+---------+----------+')
    return lines

def segment_channel(stk, ch, args, weightsdir, segmcsv):
    args.channel = ch
    args.prediction_dir = args.csv_dir = os.path.dirname(segmcsv)
    pre = RPE_Predictor(args)
    pre.cfg.class_name = ch
    pre.cfg.image_type = pre.cfg.defaultImageType(pre.cfg.class_name)
    pre.cfg.postproc = pre.AUTO_ADJUST.get(pre.cfg.class_name, imagetools.POSTPROC_NONE)
    epoch, wpath = pre.cfg.find_model_weights(weightsdir)
    if wpath is None:
        print(f'No  model weights for channel {ch}  found in {weightsdir}; skipping.')
        return
    os.makedirs(args.prediction_dir, exist_ok=True)
    pre.predict_one(stk, wpath, args.prediction_dir, csvdir=args.csv_dir)

def proc_rpe_stack(rpefpath, args):
    stk = RpeStack(rpefpath)
    args.data_dir = os.path.dirname(stk.base_dir)
    mandir = stk.subdir('Manual')
    segmdir = stk.subdir(RPE_Config.PREDICTIONS_SUBDIR)
    weightsdir = os.path.abspath(stk.subdir(args.weights))
    resdir = os.path.abspath(stk.subdir(args.save_dir))
    #
    for ch in ('Actin', 'DNA'):
        print(f'Processing channel {ch} of stack {stk.base_name}')
        tbn = stk.tname(ch, '.csv')
        mancsv = os.path.join(mandir, tbn)
        if not os.path.isfile(mancsv):
            print(f'No manual segmentation of channel {ch} found for {stk.base_name}; skipping.')
            continue
        segmcsv = os.path.join(segmdir, tbn)
        if not os.path.isfile(segmcsv) or args.force_segmentation:
            segment_channel(stk, ch, args, weightsdir, segmcsv)
        if not os.path.isfile(segmcsv):
            print(f'Failed to segment channel {ch} of {stk.base_name}; skipping.')
            continue
        r = imagetools.compare_3d_annotations(stk.width, stk.height, stk.n_frames, mancsv, segmcsv)
        os.makedirs(resdir, exist_ok=True)
        rbn = stk.tname(ch, '.txt')
        restxt = os.path.join(resdir, rbn)
        relres = os.path.join(args.save_dir, rbn)
        print(f'Write {ch} comparison results to {relres}')
        title = f'2D Comparison - {ch} of {stk.base_name}'
        lines = to_text(title, r.base_slices, r.cmp_slices, r.f_pos, r.f_neg, r.fragm, r.fused, r.pct_match)
        lines.append('')
        title = f'3D Comparison - {ch} of {stk.base_name}'
        lines.extend(to_text(title, r.base_cells, r.cmp_cells, r.f_pos_3d, r.f_neg_3d, r.fragm_3d, r.fused_3d, r.pct_match_3d))
        with open(restxt, 'wt', encoding='latin-1') as fo:
            for line in lines:
                fo.write(line+'\n')
    #

def main(arglist):
    import argparse
    
    default_wd = f'../{RPE_Config.WEIGHTS_SUBDIR}'
    default_savedir = 'EvaluationResults'

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate segmentation quality by comparing predicted annotations with manual ones.')
    parser.add_argument('rpefile', nargs='+',
            help='RPE Meta File .rpe.json, OME TIFF .ome.tif (or directory containing .rpe.json/.ome.tif)\n')
    parser.add_argument('-w', '--weights', required=False,
            metavar="/path/to/weights",
            default=default_wd,
            help='Path to directory containing .pth.\n'+\
                'Relative to the directory of input files. Default: "%s/".' % (default_wd,))
    parser.add_argument('-s', '--save-dir', required=False,
            metavar="/path/to/save/validation/results",
            default=default_savedir,
            help='Output directory, relative to the directory or input files.\nDefault: "%s"' % (default_savedir,))
    parser.add_argument('-D', '--disable-gpu', required=False, action="store_true",
            help="Disable GPU(s), e.g. if GPUs have insufficient memory and the script crashes.")
    parser.add_argument('-f', '--force-segmentation', required=False, action="store_true",
            help='Force segmentation of input stacks, even if a previous segmentation already exists\n'+\
            '(e.g. if model weights have been updated).')

    args = parser.parse_args(arglist)
    args.data_dir = os.path.abspath('.')
    if args.disable_gpu:
        print ('Disabling GPUs (if any).')
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        
    for rpefpath in iter_multi_stacks(args, recurse=False):
        proc_rpe_stack(rpefpath, args)

    return 0

if __name__ == '__main__':
    
    try:
        rc = main(sys.argv[1:])
        print(f'Done, exiting({rc}).')
    except Exception as ex:
        print('Exception:', str(ex))
        rc = -1
        raise

    sys.exit(rc)
