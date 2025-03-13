# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:14:41 2023

@author: 
"""

"""
Setup of our data sets used herein:
-------------------------------------
see routine get_data for details

General convention:
- the day a forecast was made: "forecast day" fcd
- the day forecasted dfc
- forecast = amounts for future dfc
- demand = amounts for dfc = fcd
- when we are dealing with time we assume full days only, so we compare dates instead of times

update 5-22-23:
- major misunderstanding - open amounts is to be taken literally:
    ordered amount - amounts packaged or shipped
- often, a few days prior to shipping the packaging department will start preparing the shipment
  --> the last 7-10 days prior to dfc the amounts are not accurate any more as forecasts
- on special ocassion like christmas, an end of year purge ist performed, data also not correct
- very early fc entries are per month, and not per day
- therefore, a new list is supplied with shipped amounts (also not the same as ordered!)

update 5-26-23:
- created itw_tools from it
"""                 


def show_header(data):
    """
    takes the whole data set of all data collected
    loops over all entries in our data set and print the most essential pieces of 
    information as shown below
    
    5-22-23: still old status w/o lm
    """
    print(f'show_header - products in list: {len(data)}')

    print('forecasts / open amounts:')
    #               1         2         3         4         56        7         8 
    #      12345678901234567890123456789012345678901234567890123456789012345678901234567890
    print('product          data sets   date from\tto          len from     to')
    for e in data:
        p = e['p']
        om = e['om']
        n = len(om)
        o = om[0]
        mind = o['fcd'] # datetime.strptime(d[0], '%d.%m.%Y').date()
        maxd = mind
        minl = len(o['v'])
        maxl = minl
        for o in om:
            l = len(o['v']) 
            dt = o['fcd'] # datetime.strptime(d[0], '%d.%m.%Y').date() 
            minl = minl if minl <= l else l
            maxl = maxl if maxl >= l else l
            mind = mind if mind <= dt else dt
            maxd = maxd if maxd >= dt else dt
        print(f'{p:<17}{n:<12}{mind.date()}\t{maxd.date()}\t{minl:<13}{maxl}')
            
    print('demands / shipped amounts:')
    #               1         2         3         4         56        7         8 
    #      12345678901234567890123456789012345678901234567890123456789012345678901234567890
    print('product          data sets   date from\tto          shipments  amount from  to')
    for e in data:
        p = e['p']
        lm = e['lm']
        n = len(lm)
        o = lm[0]
        mind = o['dfc'] # datetime.strptime(d[0], '%d.%m.%Y').date()
        maxd = mind
        minl = o['v']
        maxl = minl
        ns = 0
        for o in lm:
            dt = o['dfc'] # datetime.strptime(d[0], '%d.%m.%Y').date()
            l = o['v']
            minl = minl if minl <= l else l
            maxl = maxl if maxl >= l else l
            mind = mind if mind <= dt else dt
            maxd = maxd if maxd >= dt else dt
            ns = ns + 1 if l > 0 else ns
        print(f'{p:<17}{n:<12}{mind.date()}\t{maxd.date()}\t{ns:<11}{minl:<13}{maxl}')
            

def sort_all_bydate_in_place(data):
    """
    takes the whole data set of all data collected
    makes sure that for every one product the open amounts are sorted by fcd in ascending order
    no return value, sorts in place
    5-22-23: updated to new list
    """
    for e in data:
        if 'om' in e:
            e['om'] = sorted_om(e['om'])
        if 'lm' in e:
            e['lm']= sorted_lm(e['lm'])

def sorted_lm(lm):
    """
    takes the list of data dicts 
    makes sure they are sorted by fcd in ascending order
    returns a sorted list of dicts
    5-22-23: updated to new list
    """
    return sorted(lm, key = lambda x:x['dfc'])

        
def sorted_om(om):
    """
    takes the list of data dicts 
    makes sure they are sorted by fcd in ascending order
    returns a sorted list of dicts
    5-22-23: updated to new list (no change required)
    """
    return sorted(om, key = lambda x:x['fcd'])

def find_all_date_gaps(data):
    """
    takes the whole data set of all data collected
    starts an fcd gap analysis for every one entry
    5-26-23: updated to lm
    9-25-23: sort first
    """
    print('find_all_date_gaps, sort first:')
    sort_all_bydate_in_place(data)
    for e in data:
        find_date_gaps_om(e)
        find_date_gaps_lm(e)
        
def find_date_gaps_om(e):
    """
    takes one product dataset only and performes a fcd gap analysis
    prints all date gaps
    prerequisite: list must be sorted by fcd
    """
    p = e['p']
    print(f'forecasts - product {p}:')
    print('   gap between\tand')
    om = e['om']
    o = om[0]
    thisdate = o['fcd']
    for nd in om[1:]:
        nextdate = nd['fcd']
        days = distance_days(thisdate, nextdate, rounded = True) 
        if days != 1:
            print(f'   {thisdate.date()}\t{nextdate.date()}')
        thisdate = nextdate
    
def find_date_gaps_lm(e):
    """
    takes one product dataset only and performes a dfc gap analysis
    prints all date gaps
    prerequisite: list must be sorted by dfc
    """
    p = e['p']
    print(f'demands - product {p}:')
    print('   gap between\tand')
    lm = e['lm']
    o = lm[0]
    thisdate = o['dfc']
    for nd in lm[1:]:
        nextdate = nd['dfc']
        days = distance_days(thisdate, nextdate, rounded = True) 
        if days != 1:
            print(f'   {thisdate.date()}\t{nextdate.date()}')
        thisdate = nextdate

def close_all_date_gaps(data, om_approach = 'copy last and next'):
    """
    takes the whole data set of all data collected
    closes fcd gaps for every one entry
    9-25-23: first defined
    """
    print('close_all_date_gaps, sort first:')
    sort_all_bydate_in_place(data)
    for e in data:
        close_date_gaps_om(e, approach = om_approach)
        close_date_gaps_lm(e)
        
def close_date_gaps_om(e, approach = 'copy last and next'):
    """
    takes one product dataset only and closes all fcd gaps
    prints all date gaps
    prerequisite: list must be sorted by fcd
    approaches:
        'copy last': copy last values, purge front, fill end with zeroes
        'copy last and next': copy last values first, purge front, fill rear with next values
    """
    from datetime import timedelta
    p = e['p']
    print(f'forecasts - product {p}:')
    print('   gap between\tand')
    om = e['om']
    o = om[0]
    new_om = []
    thisdate = o['fcd']
    thisv = o['v']
    for nd in om[1:]:
        nextdate = nd['fcd']
        nextv = nd['v']
        days = distance_days(thisdate, nextdate, rounded = True) 
        if days != 1:
            print(f'   {thisdate.date()}\t{nextdate.date()}')
            for i in range(1, days):
                fcd = thisdate + timedelta(days = i)
                if approach == 'copy last': 
                    newv = thisv[i:] + [0 for _ in range(i)] 
                elif approach == 'copy last and next':
                    fi = len(thisv) - days # from index
                    ti = min(len(nextv), fi + i) # to index
                    if fi < len(nextv):
                        newv = thisv[i:] +  nextv[fi:ti]
                else:
                    ValueError('{approach} not a valid option')
                new_om.append({'fcd': fcd, 'v': newv})
        thisdate = nextdate
        thisv = nextv
    e['om'] += new_om
    e['om'] = sorted_om(e['om'])
    if len(new_om) > 0:
        print('closed all om date gaps')


def close_date_gaps_lm(e):
    """
    takes one product dataset only and loses all dfc gaps
    prints all date gaps
    prerequisite: list must be sorted by dfc
    approach: empty ship dates are filled with 0 demand
    """
    from datetime import timedelta
    p = e['p']
    print(f'demands - product {p}:')
    print('   gap between\tand')
    lm = e['lm']
    o = lm[0]
    new_lm = []
    thisdate = o['dfc']
    for nd in lm[1:]:
        nextdate = nd['dfc']
        days = distance_days(thisdate, nextdate, rounded = True) 
        if days != 1:
            print(f'   {thisdate.date()}\t{nextdate.date()}')
            for i in range(1, days):
                dfc = thisdate + timedelta(days = i)
                new_lm.append({'dfc': dfc, 'v': 0})
        thisdate = nextdate
    e['lm'] += new_lm
    e['lm'] = sorted_lm(e['lm'])
    if len(new_lm) > 0:
        print('closed all lm date gaps')




def get_demands(e, skip_zeros = False, min_date = None, max_date = None):
    """
    takes a data set dict for one product
    optional: skip_zeros - then only those events where demand != 0 are included
    optional: only include demands from min_date to max_date
    extracts real demands from lm
    returns
       product
       dfc list
       demand list of same size holding amounts
    6-22-23: added min and max date
    """
    p = e['p']
    # if product never shipped:
    if not 'lm' in e:
        return p, [], []

    lm = e['lm']
    t = [l['dfc'] for l in lm]
    d = [l['v'] for l in lm]

    if min_date != None:
        filtered = [(ti, di) for ti, di in zip(t, d) if ti >= min_date]
        t, d = zip(*filtered)

    if max_date != None:
        filtered = [(ti, di) for ti, di in zip(t, d) if ti <= max_date]
        t, d = zip(*filtered)
        
    if not skip_zeros: 
        return p, t, d

    # include only when demand != 0
    filtered = [(ti, di) for ti, di in zip(t, d) if di != 0]
    t, d = zip(*filtered)
    return p, t, d


def get_forecasts(e):
    """
    takes a data set dict for one product
    extracts forecasts, by definition is this anything after the 1st forecast element 
    returns
       product
       dfc list
       forecast list of same size holding lists of amounts for days +1, +2, ... 
    """
    p = e['p']
    om = e['om']
    t = [o['fcd'] for o in om]
    d = [o['v'][1:] for o in om]
    return p, t, d


def haircut_matrix(m):
    """
    takes the list of data tuples 
    truncates all embedded lists of forecasts to the same lenght
    returns a new list of tuples in rectangular form
    """
    ml = len(m[0]) # initial minimum lenght
    for i in m[1:]:
        ml = len(i) if len(i) < ml else ml
    mn = [] # new matrix
    for i in m:
        mn.append(i[0:ml])
    return mn


def all_mini_statistics_d_and_fc(d):
    """
    takes the whole data set of all data collected
    loops over all entries in our data set and print the most essential pieces of 
    statistical information as shown below
    note that treatment of demand is ok, but treatment of forecasts is not, 
    as we would have to calculate mu and sigma per fcd, but we do it for all data collected
    we are lazy and use numpy functions, for that reason data needs a haircut prior
    """
    import numpy as np
    print('all_mini_statistics_d_and_fc')
        #               1         2         3         4         56        7         8 
    #      12345678901234567890123456789012345678901234567890123456789012345678901234567890
    print('product          mu_d        vc_d     mu_fc       vc_fc    ratio_mu_d_fc  ratio_max_d_fc')

    for e in d:
        
        p, _, val = get_demands(e)   
        mu_d = np.mean(val)
        max_d = max(val)
        sigma_d = np.std(val)
        vc_d = sigma_d / mu_d
        
        _, _, val = get_forecasts(e)   
        dh = haircut_matrix(val)
        mu_fc = np.mean(dh)     
        sigma_fc = np.std(dh)
        max_fc = np.max(dh)
        vc_fc = sigma_fc / mu_fc

        ratio_max_d_fc = max_d / max_fc
        ratio_mu_d_fc = mu_d / mu_fc

        print(f'{p:<17}{mu_d:<12.4}{vc_d:<9.4}{mu_fc:<12.4}{vc_fc:<9.4}{ratio_mu_d_fc:<15.4}{ratio_max_d_fc:<12.4}')
    

def distance_days(t0, t1, rounded = True):
    """
    takes 2 datetime objects t0 and t1 and calculates the
    distance in either partial or full days between them, depending
    on whether rounded is set or no
    """
    tdist = (t1.date() - t0.date()).total_seconds() / 3600 / 24
    if rounded:
        return round(tdist)
    else:
        return tdist          


def fc_for_day(om, dfc):
    """
    takes a list of data tuples and a specific day to be forecasted
    extracts the forecast data for a given demand day dfc
    returns 
       list of fcd
       list of forecast amounts

    this one is tricky, so here is an example:       
    note open amount today is stored in row 0, FC is the data from 1 on.
    note that FC :
                  dates
            fcd   1  2  4  5  7
       today      a  b  c  d  e
       FC   +1    e  f  g  h  i
            +2    j  k  l  m  n
            +3    o  p  q  r  s
            +4    t  u  v  w  x
       dcf = 6 --> last column involved is the 4th (fcd = 5 < dcf = 6)
       The FC vector is assembled as follows:
           col 1 is omitted as the data set is not large enough:
                 fcd = 1 --> dcf = 6, distance = 5, only 4 values avail
           col 2 distance = 6 - 2 = 4 - use the 4th value as 1st entry
           col 3 distance = 2 - use 2nd value
           col 4 distance = 1 - use 1st value
           --> fc = [u, l, h]
               t  = [2, 4, 5]
    way cool! for the 1st time we can observe the fc stream for any given
    demand day 
    """
    t = []
    fc = []
    om1 = []
    for o in om:
        if o['fcd'].date() < dfc.date():
            om1.append(o)
    om1 = sorted_om(om1)
    for o in om1:
        fcd = o['fcd']
        # the distance in days is the (1) index in the vector
        dist_days = distance_days(fcd, dfc, rounded = True)         
        data = o['v'][1:]
        if len(data) >= dist_days:
            t.append(fcd)
            fc.append(data[dist_days - 1])
    return t, fc


def merged_data(e1, e2, np):
    """
    takes two separate product dicts and a new name
    integrates the two data sets by
        same fcd:
            add up demands
            copy the values of the longer list at the end
        different fcd:
            copy the the whole list 
    returns the new data dict with new name
    5-26-23: added lm, *** not tested yet ***
    """

    e = {'p': np, 'om': [], 'lm': []}
    
    # merge om:
    om1 = sorted_om(e1['om'])
    om2 = sorted_om(e2['om'])
    # 1st round: check for o1 and transfer
    for o1 in om1:
        o1notIn = True
        for o2 in om2:
            # same date - add data point by point
            if o1['fcd'] == o2['fcd']: 
                om = [x + y for x, y in zip(o1['v'], o2['v'])]
                # just in case, if one list is longer than the other ...
                if len(o1['v']) > len(o2['v']):
                    om = om + o1['v'][len(o2['v']):]
                elif len(o2['v']) > len(o1['v']):
                    om = om + o2['v'][len(o1['v']):]
                e['om'].append({'fcd': o1['fcd'], 'v': om})
                o1notIn = False
        if o1notIn:
            e['om'].append(o1)
    # 2nd round: check for missing o2 and transfer
    for o2 in om2:
        o2notIn = True
        for o1 in om1:
            o2notIn = False if o1['fcd'] == o2['fcd'] else o2notIn
        if o2notIn:
            e['om'].append(o2)
    e['om'] = sorted_om(e['om'])

    # merge lm:
    print('merged_data: lm not tested yet!')
    lm1 = sorted_lm(e1['lm'])
    lm2 = sorted_lm(e2['lm'])
    # 1st round: check for o1 and transfer
    for l1 in lm1:
        l1notIn = True
        for l2 in lm2:
            # same date - add data point by point
            if l1['dfc'] == l2['dfc']: 
                lm = l1['v'] +  l2['v']
                e['lm'].append({'dfc': l1['dfc'], 'v': lm})
                l1notIn = False
        if l1notIn:
            e['lm'].append(l1)
    # 2nd round: check for missing o2 and transfer
    for l2 in lm2:
        l2notIn = True
        for l1 in lm1:
            l2notIn = False if l1['dfc'] == l2['dfc'] else l2notIn
        if l2notIn:
            e['lm'].append(l2)
    e['lm'] = sorted_lm(e['lm'])

    return e

def get_data_for_p(d, p):
    """
    takes the whole data set of all data collected and a product name
    returns the dict holding the data for this one product only
    """
    for e in d:
        if e['p'] == p:
            return e
    raise ValueError('{p} not included in data set')
    
def find_max_index(numbers):
    """
    from ChatGPT: find index of biggest number in the list
    """
    max_value = max(numbers)
    max_index = numbers.index(max_value)
    return max_index

def list_of_fc_changes(om, first_only = False):
    """
    returns a sorted list of all fc changes in om 
    with
       fcd: the day the forecast was made
       dfc: the starting date, the gap is from this to the next day
       dv: gap in units v from dfc to the next day
    the list is sorted by dv in ascending order
    6-30-23: added switch first_only to include only the first fcd the change
             was mentioned
    """
    from datetime import timedelta
    res = []
    for o in om:
        for i, (v0, v1) in enumerate(zip(o['v'][:-1], o['v'][1:])):
            res.append({'fcd': o['fcd'], \
                        'dfc': o['fcd'] + timedelta(days = i), \
                        'dv': v1 - v0})
    if first_only:
        # the same dv event has most likely been recorded in several fc, so 
        # list the 1st incidence it was described, dfc that is
        # 1. sort list by fcd
        res = sorted(res, key = lambda x: x['fcd'])
        # 2. copy all items if the same combination of dfc and v is not yet 
        #    in the new list, automatically copying only the earlist fcd item
        res1 = []
        for r in res:
            is_in_list = False
            for r1 in res1:
                if r['dv'] == r1['dv'] and r['dfc'] == r1['dfc']:
                    is_in_list = True
                    break
            if not is_in_list:
                res1.append(r)
        res = res1
        
    res = sorted(res, key = lambda k: k['dv'], reverse = False)
    return res


def moving_average(t, v, type = 'center', num_days = 30):
    """
    takes t and v and calculates a moving month amount of v, returns time and value lists
    type: front, center, rear for time position
    asumes t and v to be of same size
    """
    import pandas as pd
    # Convert array of integers to pandas series
    ser = pd.Series(v)
    # Get the window of series of observations of specified window size
    win = ser.rolling(num_days)
    # Create a series of moving averages of each window
    moving_sum = win.mean()
    # Convert pandas series back to list
    moving_sum_list = moving_sum.tolist()
    if type == 'front':
        t1 = t[:-num_days-1]
    elif type == 'center':
        fdist = round(num_days/2)
        rdist = num_days - fdist
        t1 = t[fdist-1:-rdist]
    elif type == 'rear':
        t1 = t[num_days-1:]
    else:
        raise ValueError('{type} not a valid option')
    # Remove null entries from the list
    return t1, moving_sum_list[num_days - 1:]


def minmax_dfc(e, is_min = True):
    """
    wrapper around a combination of min_dfc_lm and min_dfc_om, provides the
    more extrem of the two
    in general finds the lowest or highest dfc among both lm and om in e
    """
    if is_min:
        return min(minmax_dfc_lm(e['lm'], is_min), minmax_dfc_om(e['om'], is_min))
    else:
        return max(minmax_dfc_lm(e['lm'], is_min), minmax_dfc_om(e['om'], is_min))

def minmax_dfc_lm(lm, is_min = True):
    if is_min:
        res = min(lm, key=lambda x: x['dfc'])
    else:
        res = max(lm, key=lambda x: x['dfc'])
    return res['dfc']

def minmax_dfc_om(om, is_min = True):
    from datetime import timedelta
    if is_min:
        res = min(om, key=lambda x: x['fcd'] + timedelta(min(len(x['v']), 1))) # note forecasts always starts the next day
        return res['fcd'] + timedelta(min(len(res['v']), 1))
    else:
        res = max(om, key=lambda x: x['fcd'] + timedelta(len(x['v'])))
        return res['fcd'] + timedelta(len(res['v']))

def no_date_gaps(l):
    """
    returns whether or no there are date gaps in the datetime list provided
    """
    from datetime import timedelta
    l.sort(key = lambda dt: dt)
    for d0, d1 in zip(l[:-1], l[1:]):
        if d1.date() - d0.date() > timedelta(days = 1):
            return False
    return True

def get_past_demand_and_future_fc_datasets(e, dim_d, dim_fc, target_day = 1, last_days_fc_ignored = 0):
    """
    provides a set of cases
    
    each case is defined as 
    . past dim_d days of actual demand stream through dfc = x
    . dim_fc stream of of forecast estimates for target_day out starting dfc = x
    . stream of outputs for dfc = x + target_day num of days
    . stream of target day dfc

    all cases in e are provided that fulfill the requirements
    . no date gaps
    . data long enough
    12.7.23 added dfcs
    """
    from datetime import timedelta
      
    past_demands = []
    fc_changes = []
    dfcs = []
    actual_demand = []
    
    _, td, vd = get_demands(e)
    
    # as we always need a demand value as final result, it is perfectly fine 
    # to loop over demand as an overall cycle
    for i in range(0,len(vd)):
        # get local demand stream if there is enough data
        if i >= dim_d:
            tdi = td[i-dim_d:i]
            vdi = vd[i-dim_d:i]
        else:
            tdi = []
            vdi = []
        # get target value, first the date
        # dfc is date of last demand + target days out, or
        # date[i=0] - 1 + target days out if there is no last demand
        if len(tdi) > 0:
            dfc = tdi[-1] + timedelta(days = target_day)
        else:
            dfc = td[i] + timedelta(days = target_day - 1)
        if no_date_gaps(tdi) and dfc in td:
            # finish up getting target data, now the value
            dfc_demand = vd[td.index(dfc)]
            # get corresponding forecast 
            tfc, vfc = fc_for_day(e['om'], dfc)
            # truncate left if there is enough data
            if len(tfc) - last_days_fc_ignored - dim_fc >= 0:
                tfc = tfc[len(tfc)-last_days_fc_ignored-dim_fc:]
                vfc = vfc[len(vfc)-last_days_fc_ignored-dim_fc:]
            else:
                tfc = []
                vfc = []
            if no_date_gaps(tfc) and len(tfc) + last_days_fc_ignored >= dim_fc:
                # truncate right
                vfc = vfc if last_days_fc_ignored <= 0 else vfc[:-last_days_fc_ignored]
            else:
                # date gaps or data not long enough
                vfc = []
            # update results if applicable
            if (dim_d > 0 and len(vdi) == 0) or (dim_fc > 0 and len(vfc) == 0):
                pass
            else:
                past_demands.append(vdi)
                fc_changes.append(vfc)
                actual_demand.append(dfc_demand)
                dfcs.append(dfc)
    return past_demands, fc_changes, actual_demand, dfcs


    
def floatanize(t):
    """
    takes a tensor and converts every single item in the given tensor into float
    returns the result
    """
    if type(t) == list:
        res = []
        for ti in t:
            res.append(floatanize(ti))
        return res
    else:
        return float(t)
    

if __name__ == '__main__':

    import pickle

    with open("itw_d.pkl", "rb") as f:
        d = pickle.load(f)
            
    if False:
        t = floatanize([[1, 2, 3], [4, 5, 6]])
        
    if True:
        e = d[0]
        i_d, i_fc, o, dfcs = get_past_demand_and_future_fc_datasets(e, dim_d = 40, dim_fc = 30, \
                            target_day = 1, last_days_fc_ignored = 10)
        
    if False:
        p, t, v = get_demands(d[0])
        t1, v1 = moving_average(t, v)
        
    if False:
        show_header(d)
        find_all_date_gaps(d)
        all_mini_statistics_d_and_fc(d)

    if False:        
        dm = []
        dm.append(merged_data(get_data_for_p(d, '10512005001'), get_data_for_p(d, '10512005001 8'), '10512005001 ges'))
        dm.append(merged_data(get_data_for_p(d, '10397000001'), get_data_for_p(d, '10397000001 1'), '10397000001 ges'))
        dm.append(merged_data(get_data_for_p(d, '10397000001'), get_data_for_p(d, '10397000001 2'), '10397000001 ges'))
        dm.append(merged_data(get_data_for_p(d, '10416000001'), get_data_for_p(d, '10416000001 2'), '10416000001 ges'))
        show_header(dm)
        find_all_date_gaps(dm)
        all_mini_statistics_d_and_fc(dm)
     
