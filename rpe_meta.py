__all__ = ('meta_from_pathname', 'meta_preview', 'metas_from_basedir', 'generate_metas', 'copy_meta')

import os, sys, datetime, glob
import shutil
from collections import defaultdict
import json

def parse_week(s):
    s = s.upper()
    if s.startswith('WK'):
        rest = s[2:]
    elif s.startswith('W'):
        rest = s[1:]
    else:
        return None
    try:
        return int(rest)
    except Exception:
        return None
    
def parse_alphanum(s, a):
    parts = s.upper().split(a)
    if len(parts) < 2:
        return None
    r = ''
    for c in parts[1]:
        if not c in '0123456789': break
        r = r + c
    try:
        return int(r)
    except Exception:
        return None
    
def fix_week(fmeta, week):
    if not fmeta or not 'basename' in fmeta:
        return
    basename = fmeta['basename']
    parts = basename.split('-')
    for i, p in enumerate(parts):
        if parse_week(p) is None: continue
        if p.upper() != week:
            parts[i] = week
            fmeta['basename'] = '-'.join(parts)
        break
    else:
        fmeta['basename'] = week + '-' + basename
        
def fix_basename(fmeta, mesmeta):
    if not fmeta or not mesmeta or not 'basename' in fmeta:
        return
    basename = fmeta['basename']
    # 01-22-19_18-19-07_D02_T0001F001L01A01Z02C02.tif
    try:
        dt = datetime.datetime.strptime(basename[:18], '%m-%d-%y_%H-%M-%S_')
        basename = basename[18:]
    except Exception:
        pass
    parts = []
    if 'plate' in mesmeta:
        parts.append(mesmeta['plate'])
    if 'week' in mesmeta:
        parts.append(mesmeta['week'])
    if 'Organelle' in mesmeta:
        parts.append(mesmeta['Organelle'])
    if len(parts) > 0:
        basename = '-'.join(parts) + '_' + basename
    fmeta['basename'] = basename

def parse_file_name(fn):
    res = {}
    bname, ext = os.path.splitext(fn)
    if ext.lower() not in ('.tif', '.tiff'):
        return None
    res['ext'] = ext
    parts = bname.split('_')
    if len(parts) < 2:
        return None
    suff = parts[-1]
    frame = parse_alphanum(suff, 'Z')
    chan = parse_alphanum(suff, 'C')
    fov = parse_alphanum(suff, 'F')
    if frame is None or chan is None:
        return None
    basename = '_'.join(parts[:-1])
    if not fov is None:
        basename = basename + '_F%03d' % (fov,)
        res['fov'] = fov
    res['basename'] = basename
    res['frame'] = frame
    res['channel'] = chan
    if len(parts) > 2:
        res['well'] = parts[-2]
    return res

def parse_dirname(dfn):
    week = None
    res = {}
    for pp in dfn.replace('_', '-').split('-'):
        ww = parse_week(pp)
        if not ww is None:
            week = pp.upper()
            res['week'] = week
        pl = parse_alphanum(pp, 'P')
        if not pl is None:
            res['plate'] = 'P%d' % (pl,)
    if week is None:
        return None
    return res

def parse_mes(fpath):
    bname, ext = os.path.splitext(os.path.basename(fpath))
    if ext.lower() != '.mes':
        return None
    try:
        parts = bname.split('-')
        w = parse_week(parts[-1])
        if w is None:
            return None
        mesmeta = { 'week' : 'W%d' % (w,), 'Organelle' : parts[-2] }
    except Exception:
        return None
    return mesmeta

def fmeta_to_jsonable(fmeta_list, basename):
    chans = set()
    frames = set()
    page_map = {}
    ch0 = None
    mes = None
    for fmeta in fmeta_list:
        chan = fmeta['channel']
        frame = fmeta['frame']
        chans.add(chan)
        frames.add(frame)
        page_map[(chan, frame)] = fmeta['rpath']
        if ch0 is None and 'Organelle' in fmeta:
            ch0 = fmeta['Organelle']
        if mes is None and 'mes' in fmeta:
            mes = fmeta['mes'].replace('\\', '/')
    chans = sorted(chans)
    if len(chans) > 4:
        #print ('Reducing #channels from %d to 4' % (len(chans),))
        chans = chans[0:4]
    frames = sorted(frames)
    flist = []
    for fr in frames:
        for ch in chans:
            rpath = page_map.get((ch, fr), '')
            flist.append(rpath.replace('\\', '/'))
    res = {
        'basename': basename,
        'n_channels': len(chans),
        'n_frames': len(frames),
        'filelist': flist,
        'dimorder': 'XYCZ',
    }
    if not mes is None:
        res['mes'] = mes
    if len(chans) == 4:
        if ch0 is None:
            try:
                # Decode channel 0 label from basename (such as P1-W3-ZO1_D04_F004 -> Z01)
                suff = basename.split('-')[-1]
                ch0 = suff.split('_')[0].upper()
            except Exception:
                ch0 = 'GFP'
        res['labels'] = [ch0, 'DNA', 'Actin', 'Membrane']
    return res

def meta_from_pathname(fpath):
    fdir, fn = os.path.split(fpath)
    fmeta0 = parse_file_name(fn)
    if fmeta0 is None:
        return None
    basedir, subdir = os.path.split(fdir)
    week = None
    plate = None
    dmeta = parse_dirname(subdir)
    if not dmeta is None:
        week = dmeta['week']
        plate = dmeta.get('plate', None)
    basename0 = fmeta0['basename']
    rel = subdir
    org = None
    mes = None
    mesmeta = None
    if week is None:
        # Try multi-row naming convention (Organelle comes from *.mes)
        basedir, subdir2 = os.path.split(basedir)
        dmeta = parse_dirname(subdir2)
        if dmeta is None:
            return None
        week = dmeta['week']
        plate = dmeta.get('plate', None)
        for mes in glob.glob(os.path.join(fdir, '*.mes')):
            mesmeta = parse_mes(mes)
            if not mesmeta is None:
                break
        else:
            return None
        org = mesmeta['Organelle']
        mesmeta['absdir'] = fdir
        mesmeta['reldir'] = rel = os.path.join(subdir2, subdir)
        mesmeta['mes'] = mes = os.path.relpath(mes, basedir)
        if not plate is None:
            mesmeta['plate'] = plate
    #
    fmetas = []
    basename = None
    for fn in os.listdir(fdir):
        fpath = os.path.join(fdir, fn)
        if not os.path.isfile(fpath):
            continue
        fmeta = parse_file_name(fn)
        if fmeta is None:
            continue
        if fmeta['basename'] != basename0:
            continue
        if mesmeta is None:
            fix_week(fmeta, week)
        else:
            fix_basename(fmeta, mesmeta)
            fmeta['Organelle'] = org
            fmeta['mes'] = mes
        fmeta['rpath'] = os.path.join(rel, fn)
        if basename is None:
            basename = fmeta['basename']
        fmetas.append(fmeta)
    if basename is None:
        return None
    res = fmeta_to_jsonable(fmetas, basename)
    res['basedir'] = basedir
    res['basename'] = basename
    # res['reldir'] = rel
    return res

def _meta_to_text(fmeta, sep='\n'):
    labels = [('Organelle', 'Organelle'), ('plate', 'Plate'), ('week', 'Week'), ('well', 'Well'),
              ('fov', 'Field of view')]
    parts = []
    if 'basename' in fmeta:
        # Ex. P1-W1-ZO1_D07_F001.rpe.json
        parts.append('Meta-file: %s.rpe.json' % (fmeta['basename'],))
    for key, lab in labels:
        if key in fmeta:
            parts.append('%s: %s' % (lab, fmeta[key]))
    return sep.join(parts)

def meta_preview(fpath):
    fdir, fn = os.path.split(fpath)
    fmeta = parse_file_name(fn)
    if fmeta is None:
        return None
    mesmeta = None
    for mes in glob.glob(os.path.join(fdir, '*.mes')):
        mesmeta = parse_mes(mes)
        if not mesmeta is None:
            fmeta['Organelle'] = mesmeta['Organelle']
            fmeta['week'] = mesmeta['week']
            break
    basedir, subdir = os.path.split(fdir)
    rel = subdir
    dmeta = parse_dirname(subdir)
    if dmeta is None:
        if not mesmeta is None:
            basedir, subdir2 = os.path.split(basedir)
            rel = os.path.join(subdir2, subdir)
            dmeta = parse_dirname(subdir2)
            if not dmeta is None:
                fix_basename(fmeta, mesmeta)
    else:
        fix_week(fmeta, dmeta['week'])
    if not dmeta is None:
        fmeta['week'] = dmeta['week']
        if 'plate' in dmeta:
            fmeta['plate'] = dmeta['plate']
        fmeta['basedir'] = basedir
        # fmeta['reldir'] = rel
        fmeta['text'] = _meta_to_text(fmeta)
    return fmeta

def metas_from_basedir(basedir, verbose=False):
    wdata = defaultdict(list)
    
    for dfn in os.listdir(basedir):
        dpath = os.path.join(basedir, dfn)
        if not os.path.isdir(dpath): continue
        dmeta = parse_dirname(dfn)
        if dmeta is None:
            continue
        week = dmeta['week']
        
        subdirs = []
        if verbose:
            print ('Reading: ' + dpath)
        for fn in os.listdir(dpath):
            fpath = os.path.join(dpath, fn)
            if os.path.isdir(fpath):
                for mes in glob.glob(os.path.join(fpath, '*.mes')):
                    mesmeta = parse_mes(mes)
                    if not mesmeta is None:
                        if 'plate' in dmeta:
                            mesmeta['plate'] = dmeta['plate']
                        mesmeta['absdir'] = os.path.join(dpath, fn)
                        mesmeta['reldir'] = os.path.join(dfn, fn)
                        mesmeta['mes'] = os.path.relpath(mes, basedir)
                        # print(mes, mesmeta)
                        subdirs.append(mesmeta)
                        break
                continue
            fmeta = parse_file_name(fn)
            if fmeta is None: continue
            fix_week(fmeta, week)
            fmeta['rpath'] = os.path.join(dfn, fn)
            wdata[fmeta['basename']].append(fmeta)
            
        for mesmeta in subdirs:
            dpath = mesmeta['absdir']
            if verbose:
                print ('Reading: '+dpath)
            week = mesmeta['week']
            org = mesmeta['Organelle']
            rel = mesmeta['reldir']
            mes = mesmeta['mes']
            # print (week, org, rel)
            for fn in os.listdir(dpath):
                fmeta = parse_file_name(fn)
                if fmeta is None: continue
                fix_basename(fmeta, mesmeta)
                fmeta['rpath'] = os.path.join(rel, fn)
                fmeta['Organelle'] = org
                fmeta['mes'] = mes
                wdata[fmeta['basename']].append(fmeta)
    #
    res = [fmeta_to_jsonable(fmeta_list, basename) for basename, fmeta_list in wdata.items()]
    return res

def generate_metas(rootdir, verbose=False):
    subdirs = []
    nmetas = 0
    cnt = 0
    for fn in os.listdir(rootdir):
        fpath = os.path.join(rootdir, fn)
        if os.path.isdir(fpath):
            subdirs.append(fpath)
            continue
        bn, ext = os.path.splitext(fn)
        if ext.lower() in ('.tif', '.tiff'):
            mp = meta_preview(fpath)
            if mp and 'basedir' in mp:
                bd = mp['basedir']
                if verbose: print(f'--- {bd} ---')
                for meta in metas_from_basedir(bd):
                    basename = meta.pop('basename')
                    jpath = os.path.join(bd, basename+'.rpe.json')
                    if verbose: print(jpath)
                    with open(jpath, 'w') as fo:
                        json.dump(meta, fo, indent=2)
                        nmetas += 1
                return nmetas
            cnt += 1
            if cnt > 20:
                return nmetas
    for subdir in subdirs:
        nmetas += generate_metas(subdir, verbose)
    return nmetas

def copy_meta(meta_fpath, tgt_dir):
    src_dir, meta_fn = os.path.split(meta_fpath)
    tgt_meta_fpath = os.path.join(tgt_dir, meta_fn)
    with open(meta_fpath, 'r') as fi:
        obj = json.load(fi)
    dirset = set()
    dirset.add(tgt_dir)
    cp_flist = []
    if 'filelist' in obj:
        for rel_path in obj['filelist']:
            src_path = os.path.join(src_dir, rel_path)
            tgt_path = os.path.join(tgt_dir, rel_path)
            dirset.add(os.path.dirname(tgt_path))
            cp_flist.append((src_path, tgt_path))
    if 'mes' in obj:
        rel_path = obj['mes']
        src_path = os.path.join(src_dir, rel_path)
        tgt_path = os.path.join(tgt_dir, rel_path)
        dirset.add(os.path.dirname(tgt_path))
        cp_flist.append((src_path, tgt_path))
    cp_flist.append((meta_fpath, tgt_meta_fpath))
    for tdir in dirset:
        if not os.path.isdir(tdir):
            os.makedirs(tdir)
    for src, tgt in cp_flist:
        shutil.copy(src, tgt)
    return len(cp_flist)

# Handle naming conventions:
#
#    <basedir>/*-W?-*/P?-W?-<Organelle>_<well>_T0001F00?L01A0?Z??C0?.tif
#
#    <basedir>/*-W?-*/<YY-MM-DD_hh-mm-ss>/*-<Organelle>-W?.mes
#                                        /*_<well>_T0001F00?L01A0?Z??C0?.tif
#
if __name__ == '__main__':
    
    cmd = None
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        args = sys.argv[2:]
        if cmd.startswith('gen'):
            if len(args) > 0:
                rootdir = args[0]
                print('Generating metas in {rootdir} ...')
                cnt = generate_metas(rootdir, verbose=True)
                print(f'Generated {cnt} meta file(s).')
            else:
                cmd = None
        elif cmd == 'copy':
            if len(args) > 1:
                tgtdir = args[0]
                for mfpath in args[1:]:
                    print(f'{mfpath} -> {tgtdir}')
                    copy_meta(mfpath, tgtdir)
            else:
                cmd = None
        else:
            cmd = None
    if cmd is None:
        print("""To generate .rpe.json metafiles in the <rootdir> and its sub-directories:

    gen[erate] <rootdir> -- generate .rpe.json metafiles in the <rootdir> and its sub-directories

To copy image data to another directory using .rpe.json metafile(s):

    copy <tgtdir> <path/to/meta> [<path/to/meta2> <path/to/meta3> ...]
""")
    
    sys.exit(0)

