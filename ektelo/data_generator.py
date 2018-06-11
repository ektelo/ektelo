""" Module to create synthetic data """

from __future__ import division
from __future__ import print_function

from builtins import zip
from builtins import range
import numpy
import scipy.stats as sci
from functools import reduce

def make_array(sample_size, universe_size, sparsity_fraction=0.8, power=1.2, prng=numpy.random):
    """ Generates a random array of counts with sparsity

    Inputs:
        sample_size:  the desired population size
        universe_size: the size of the array
        sparsity_fraction: the fraction of cells that are 0
        power: a shape parameter for the distribution. Should be > 1
        prng: random number generator like numpy.random

    Output:
        data: an array of counts. The non-zero entries come from a sample_size samples
            from a zipf distribution with parameter power, conditioned on the fact that
            only round((1-sparsity_fraction) * universe_size) values are possible.

    """
    assert power > 1
    domain = list(range(universe_size))
    non_sparse = max(1, int(round(universe_size * (1-sparsity_fraction))))
    non_sparse_domain = prng.choice(domain, size=non_sparse, replace=False)
    #print(non_sparse_domain)
    data = numpy.zeros(shape=universe_size)
    non_sparse_probs = numpy.zeros(shape=non_sparse)
    for index in range(non_sparse_domain.size):
        pmf = sci.zipf.pmf(index+1, power)
        non_sparse_probs[index] = pmf
    non_sparse_probs = non_sparse_probs/non_sparse_probs.sum()
    multi = prng.multinomial(sample_size, non_sparse_probs)
    for index in range(non_sparse_domain.size):
        translated = non_sparse_domain[index]
        data[translated] = multi[index]
    return data


def make_hist(sample_size=312467538, shape=None, s_frac=0.8, power=1.2, prng=numpy.random):
    """ Generates a random table with sparsity

    Inputs:
        sample_size:  the desired population size
        shape: a numpy tuple of the desired data shape or None
        s_frac: the fraction of cells that are 0
        power: a shape parameter for the distribution. Should be > 1
        prng: random number generator like numpy.random

    Output:
        data: an table of counts. The table has the prescribed shape. The non-zero entries
            come from a sample_size samples
            from a zipf distribution with parameter power, conditioned on the fact that
            only round((1-sparsity_fraction) * universe_size) values are possible.
        names: names of the dimensions of the data

    """
    if shape is None:
        shape = (17, 2, 115, 2, 63)
    names = None
    if shape == (17, 2, 115, 2, 63):
        names = ["relation", "sex", "age", "hispanic", "race"]
    else:
        names = ["col_%d" % i for i in range(len(shape))]
    array_data = make_array(sample_size, numpy.prod(shape), sparsity_fraction=s_frac, power=power, prng = prng)
    mydata = array_data.reshape(shape)
    return mydata, names

def make_hist_using_factfinder(prng=numpy.random):
    #SF1 table P4 field P0040001
    sample_size = 308745538 # from https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=DEC_10_SF1_P4&prodType=table

    #SF1 table P4 fields P004003, P004002
    hist_hispanic = [50477594, 258267944] # from https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=DEC_10_SF1_P4&prodType=table
    #SF1 table p8 https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=DEC_10_SF1_P8&prodType=table
    hist_race = [223553265, #White alone
                  38929319, #Black or African American alone
                   2932248, #American Indian and Alaska Native alone
                  14674252, #Asian alone
                    540013, #Native Hawaiian and Other Pacific Islander alone
                  19107368, #Some Other Race alone
                   1834212, #White; Black or African American
                   1432309, #White; American Indian and Alaska Native
                   1623234, #White; Asian
                    169991, #White; Native Hawaiian and Other Pacific Islander
                   1740924, #White; Some Other Race
                    269421, #Black or African American; American Indian and Alaska Native
                    185595, #Black or African American; Asian
                     50308, #Black or African American; Native Hawaiian and Other Pacific Islander
                    314571, #Black or African American; Some Other Race
                     58829, #American Indian and Alaska Native; Asian
                     11039, #American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander
                    115752, #American Indian and Alaska Native; Some Other Race
                    165690, #Asian; Native Hawaiian and Other Pacific Islander
                    234462, #Asian; Some Other Race
                     58981, #Native Hawaiian and Other Pacific Islander; Some Other Race
                    230848, #White; Black or African American; American Indian and Alaska Native
                     61511, #White; Black or African American; Asian
                      9245, #White; Black or African American; Native Hawaiian and Other Pacific Islander
                     46641, #White; Black or African American; Some Other Race
                     45960, #White; American Indian and Alaska Native; Asian
                      8656, #White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander
                     30941, #White; American Indian and Alaska Native; Some Other Race
                    143126, #White; Asian; Native Hawaiian and Other Pacific Islander
                     35786, #White; Asian; Some Other Race
                      9181, #White; Native Hawaiian and Other Pacific Islander; Some Other Race
                      9460, #Black or African American; American Indian and Alaska Native; Asian
                      2142, #Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander
                      8236, #Black or African American; American Indian and Alaska Native; Some Other Race
                      7295, #Black or African American; Asian; Native Hawaiian and Other Pacific Islander
                      8122, #Black or African American; Asian; Some Other Race
                      4233, #Black or African American; Native Hawaiian and Other Pacific Islander; Some Other Race
                      3827, #American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander
                      3785, #American Indian and Alaska Native; Asian; Some Other Race
                      2000, #American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race
                      5474, #Asian; Native Hawaiian and Other Pacific Islander; Some Other Race
                     19018, #White; Black or African American; American Indian and Alaska Native; Asian
                      2673, #White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander
                      8757, #White; Black or African American; American Indian and Alaska Native; Some Other Race
                      4852, #White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander
                      2420, #White; Black or African American; Asian; Some Other Race
                       560, #White; Black or African American; Native Hawaiian and Other Pacific Islander; Some Other Race
                     11500, # White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander
                      1535, #White; American Indian and Alaska Native; Asian; Some Other Race
                       454, #White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race
                      3486, #White; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race
                      1011, #Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander
                       539, #Black or African American; American Indian and Alaska Native; Asian; Some Other Race
                       212, #Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race
                       574, #Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race
                       284, #American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race
                      6605, #White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander
                      1023, #White; Black or African American; American Indian and Alaska Native; Asian; Some Other Race
                       182, #White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race
                       268, #White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race
                       443, #White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race
                        98, #Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race
                       792, #White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race
                ]
    #table P12 fields P0120002 and P0120026
    hist_sex = [151781326, 156964212] # from https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=DEC_10_SF1_P12&prodType=table
    #from https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=DEC_10_SF1_QTP2&prodType=table
    hist_age = [
        3944153,
        3978070,
        4096929,
        4119040,
        4063170,

        4056858,
        4066381,
        4030579,
        4046486,
        4148353,

        4172541,
        4114415,
        4106243,
        4118013,
        4165982,

        4242820,
        4316139,
        4395295,
        4500855,
        4585234,

        4519129,
        4354294,
        4264642,
        4198571,
        4249363,

        4262350,
        4152305,
        4248869,
        4215249,
        4223076,

        4285668,
        3970218,
        3986847,
        3880150,
        3839216,

        3956434,
        3802087,
        3934445,
        4121880,
        4364796,

        4383274,
        4114985,
        4076104,
        4105105,
        4211496,

        4508868,
        4519761,
        4535265,
        4538796,
        4605901,

        4660295,
        4464631,
        4500846,
        4380354,
        4291999,

        4254709,
        4037513,
        3936386,
        3794928,
        3641269,

        3621131,
        3492596,
        3563182,
        3483884,
        2657131,

        2680761,
        2639141,
        2649365,
        2323672,
        2142324,

        2043121,
        1949323,
        1864275,
        1736960,
        1684487,

        1620077,
        1471070,
        1455330,
        1400123,
        1371195,

        1308511,
        1212865,
        1161421,
        1074809,
        985721,

        914723,
        814211,
        712908,
        640619,
        537998,

        435563,
        344987,
        281389,
        216978,
        169449,

        129717,
        95223,
        68138,
        45900,
        32266,

        #100 - 104: 49141
        #105 - 109: 3893
        #110 +    : 330
        #will be filled in later
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0   #count for 114+
    ]
    #from https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=DEC_10_SF1_QTP2&prodType=table
    age100 = 49141 #100 - 104: 49141
    age105 = 3893 #105 - 109: 3893
    age110 = 330 #110 +    : 330
    hist_age[100:105] = fill(5,age100, age105/float(age100))
    hist_age[105:110] = fill(5,age105, age110/float(age105))
    hist_age[110:115] = fill(5,age110, age110/float(age105))

    #numbers from https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=DEC_10_SF1_P29&prodType=table
    #nonrelatives from https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=DEC_10_SF1_PCT19&prodType=table
    hist_relation = [
        77538296 + 18459253 + 20718743, #Householder
        56510377, #Spouse
        82582058, #Biological Child
         2072312, #Adopted Child
         4165886, #Stepchild
         7139601, #Grandchild
         3433951, #Brother/Sister
         3033003, #Parent
          925713, #Parent-in-law
         1216299, #Son/Daughter-in-law
         4662672, #Other relative
          770254 + 755956, #Roomer, Boarder
         1117184 + 4106181, #Housemate, Roommate
         3531271 + 4213440, #Unmarried Partner
         2290502 + 1515263, #Other Non-relative"
         3993659, #institutionalized
         3993664 # non-institutionalized
    ]
    fields = (hist_relation, hist_sex, hist_age, hist_hispanic, hist_race)
    shape = tuple(len(h) for h in fields)
    names = ["relation", "sex", "age", "hispanic", "race"]
    #assert all totals add up
    assert sample_size == sum(hist_relation)
    assert sample_size == sum(hist_sex)
    assert sample_size == sum(hist_age)
    assert sample_size == sum(hist_hispanic)
    assert sample_size == sum(hist_race)
    p = numpy.zeros(shape)
    myiter = numpy.nditer(p, flags=['multi_index'])
    while not myiter.finished:
        ind = myiter.multi_index
        p[ind] = reduce(lambda x,y: x * y[1][y[0]]/float(sample_size), list(zip(ind,fields)),1.0)
        myiter.iternext()
    p = p/p.sum() # just in case numerical issues come up
    outdata = prng.multinomial(sample_size, p.reshape(p.size))
    return outdata.reshape(p.shape), names


def fill(num, totcount, p):
    """ return an array of size num of exponentially decreasing counts
    with ratio p, total totcount

    Args:
        num: size of array to return
        totcount: desired sum of array
        p: discounting fraction

    """
    powers = [p**x for x in range(num)]
    tmp_tot = float(sum(powers))
    counts = [int(numpy.floor(totcount * the_p /tmp_tot)) for the_p in powers]
    residual = totcount - sum(counts)
    counts[0] = counts[0] + residual
    return counts

    
