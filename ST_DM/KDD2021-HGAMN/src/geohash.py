import spatial_pyso as spatial
calc_mc_geohash = spatial.calc_mc_geohash

base32_codes = [
    '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',
    'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def get_bin(dec_num):
    b = [0] * 5 #dec_num <= 31
    s = dec_num
    i = 4
    while s > 0:
        b[i] = s % 2
        s = s // 2
        i -= 1
    return b

geohash_id = {}
for i in range(len(base32_codes)):
    geohash_id[base32_codes[i]] = get_bin(i)

def get_xy(coor):
    if len(coor) < 1:
        return None
    if coor == 'NULL':
        return None
    xy = coor.split(',') #(12003.21, 23934.2)
    if len(xy) != 2:
        return None
    x = xy[0]
    y = xy[1]
    x = x.strip()
    y = y.strip()
    if len(x) < 3 or len(y) < 3:
        return None
    x = x[1:] if x[0] == '(' else x
    y = y[:-1] if y[-1] == ')' else y
    return x, y

def geohash_to_bits(geohash):
    """
        w4gx223 -> 00001 11001 01101 ...
    """
    if len(geohash) < 1:
        return [0] * 40
    bits = []
    for c in geohash:
        bits.extend(geohash_id[c])
    return bits

def get_loc_bound_gh(loc, bound):
    loc_gh = ""
    loc_xy = get_xy(loc)
    if loc_xy is not None:
        loc_gh = calc_mc_geohash(float(loc_xy[0]), float(loc_xy[1]))

    bound_xy = bound.split(';')
    if len(bound_xy) < 2:
        left_bottom, right_top = "", ""
    else:
        left_bottom, right_top = bound_xy[0], bound_xy[1] #(12952507,4816736;12954174,4819699)
    bound_gh = ""
    left_bottom_xy = get_xy(left_bottom)
    right_top_xy = get_xy(right_top)
    if left_bottom_xy is not None and right_top_xy is not None:
        #bound center point
        bound_xy = [0, 0]
        bound_xy[0] = (float(left_bottom_xy[0]) + float(right_top_xy[0])) / 2.0
        bound_xy[1] = (float(left_bottom_xy[1]) + float(right_top_xy[1])) / 2.0
        bound_gh = calc_mc_geohash(bound_xy[0], bound_xy[1])
    return geohash_to_bits(loc_gh) + geohash_to_bits(bound_gh)

def get_loc_gh(loc):
    loc_gh = ""
    loc_xy = get_xy(loc)
    if loc_xy is not None:
        loc_gh = calc_mc_geohash(float(loc_xy[0]), float(loc_xy[1]))
    return geohash_to_bits(loc_gh)


