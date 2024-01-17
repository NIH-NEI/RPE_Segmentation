
def slice_area(asize, tsize, minovl=200):
    aw = int(asize[1])
    ah = int(asize[0])
    w = int(tsize[1])
    h = int(tsize[0])
    #
    res = []
    if aw < w or ah < h:
        return res
    if tsize[0] <= 0 or tsize[1] <= 0:
        res.append((0, aw-1, 0, ah-1))
        return res
    #
    nxt = aw // w + 1
    xovl = (nxt*w - aw) // (nxt - 1)
    while xovl < minovl:
        nxt += 1
        xovl = (nxt*w - aw) // (nxt - 1)
    #
    nyt = ah // h + 1
    yovl = (nyt*h - ah) // (nyt - 1)
    while yovl < minovl:
        nyt += 1
        yovl = (nyt*h - ah) // (nyt - 1)
    #
    hw = w//2
    hh = h//2
    xmax = aw - 1
    ymax = ah - 1
    #
    xstep = aw // nxt
    ystep = ah // nyt
    #
    for yc in range(nyt):
        y0 = yc * ystep + ystep//2 - hh
        if y0 < 0: y0 = 0
        y1 = y0 + h - 1
        if y1 > ymax:
            y1 = ymax
            y0 = y1 - h + 1
        for xc in range(nxt):
            x0 = xc * xstep + xstep//2 - hw
            if x0 < 0: x0 = 0
            x1 = x0 + w - 1
            if x1 > xmax:
                x1 = xmax
                x0 = x1 - w + 1
            res.append((x0, x1, y0, y1))
    return res
