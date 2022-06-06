import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
import corner
from graphs.models.flows import MAF
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
model = MAF(3, 3, 1024, 10, 'relu')
state_dict = torch.load('/home-4/wwong24@jhu.edu/MachineLearning/NormalizingFlow/experiments/Cosmic_m1m2z_3hyper_smallz/checkpoints/model_best.pth.tar')['state_dict']
model.load_state_dict(state_dict)
model.cuda()
model.eval()
model.log_probs(torch.from_numpy(np.array([[1,2,3]])).float().cuda())
obser_max = np.array([95.74509802, 84.24619469,  1.19999985])
obser_min = np.array([2.61736694,  2.5005073 , -0.21434554])
hyper_max = np.array([5,4,1]) 
hyper_min = np.array([0.25,0.5,0])
Jacob = np.prod(obser_max-obser_min)

O12 = h5py.File('/home-4/wwong24@jhu.edu/Gwave/O3prep/injections_O1O2an_spin.h5','r')
O3 = h5py.File('/home-4/wwong24@jhu.edu/Gwave/O3prep/o3a_bbhpop_inj_info.hdf','r')
O3_selection= (O3['injections/ifar_gstlal'][()]>1) | (O3['injections/ifar_pycbc_bbh'][()]>1) | (O3['injections/ifar_pycbc_full'][()]>1)
m1 = np.append(O12['mass1_source'][()],O3['injections/mass1_source'][()][O3_selection])
m2 = np.append(O12['mass2_source'][()],O3['injections/mass2_source'][()][O3_selection])
z = np.append(O12['redshift'][()],O3['injections/redshift'][()][O3_selection])
s1z = np.append(O12['spin1z'][()],O3['injections/spin1z'][()][O3_selection])
s2z = np.append(O12['spin2z'][()],O3['injections/spin2z'][()][O3_selection])
pdraw = np.append(O12['sampling_pdf'][()],O3['injections/sampling_pdf'][()][O3_selection])
pdraw = pdraw[(s1z>-0.5)*(s1z<0.5)*(s2z>-0.5)*(s2z<0.5)]
pdraw = pdraw#*get_pchieff(chi_eff,m2/m1)
samples = torch.tensor((np.array([m1,m2,z]).T-obser_min)/(obser_max-obser_min))[(s1z>-0.5)*(s1z<0.5)*(s2z>-0.5)*(s2z<0.5)]
Ndraw = O3.attrs['total_generated']+7.1*1e7

def get_selection(hyp):
  with torch.no_grad():
    a = model.log_probs(samples.float().cuda(),torch.tensor([hyp]).float().cuda()).T[0].cpu().numpy()-np.log(Jacob)
  return np.exp(np.logaddexp.reduce(a[np.isfinite(a)]-np.log(pdraw[np.isfinite(a)]))-np.log(Ndraw))

hyp = []
N_sample = 10
for i in np.linspace(0.,1,N_sample):
	for j in np.linspace(0.,1,N_sample):
		for k in np.linspace(0,1,N_sample):
			hyp.append([i,j,k])

selection_bias = []
for i in hyp:
  selection_bias.append(get_selection(i))

hyper = hyp*(hyper_max-hyper_min)+hyper_min
fraction = np.array(selection_bias)
np.savez('/home-4/wwong24@jhu.edu/scratch/O3prep/Cosmic/misc/reweight/Cosmic_GWTC12_selection.npz',fraction=fraction,hyper=hyper)
