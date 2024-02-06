import numpy
import hvwfg

def cal_ps_hv(pf, pf_num, ref, ideal=None):
    if ideal is None and ref.shape[0] == 2:
        ideal = numpy.array([0, 0])
    elif ideal is None and ref.shape[0] == 3:
        ideal = numpy.array([0, 0, 0])
    batch_size = pf.shape[0]
    
    hvs = numpy.zeros([batch_size, 1])
    specific_hv = numpy.zeros([batch_size,1])
    ref_region = 1
    for i in range(ref.shape[0]):
        ref_region = ref_region * (ref[i] - ideal[i])
    for k in range(batch_size):
        num = pf_num[k]
        hv = hvwfg.wfg(pf[k][:num].astype(float), ref.astype(float))
        specific_hv[k] = hv
        hv = hv / ref_region
        hvs[k] = hv

    return specific_hv,hvs
